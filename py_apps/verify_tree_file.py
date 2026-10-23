#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from pathlib import Path
import struct
import sys
from tree_reference import read_particles, check_conservation, check_cpu_particles

MAGIC = b"PSFCTREE"
VERSION = 1
HEADER = struct.Struct("<8sIIQ")
SECTION_HEADER = struct.Struct("<iQQ")
NODE_RECORD = struct.Struct("<QiiiiiBiQB")


@dataclass(frozen=True)
class Node:
    key: int
    owner: int
    level: int
    parent: int
    first_child: int
    next_sibling: int
    child_mask: int
    particle_begin: int
    particle_count: int
    flags: int

    @property
    def is_leaf(self):
        return bool(self.flags & 1)

    @property
    def children_available(self):
        return bool(self.flags & 2)

    @property
    def is_branch(self):
        return bool(self.flags & 4)

    @property
    def is_remote(self):
        return bool(self.flags & 8)


def read_exact(stream, size, description):
    data = stream.read(size)
    if len(data) != size:
        raise ValueError(f"truncated {description}: expected {size} bytes, got {len(data)}")
    return data


def read_tree_file(path):
    with path.open("rb") as stream:
        magic, version, tree_count, total_particles = HEADER.unpack(
            read_exact(stream, HEADER.size, "header")
        )
        if magic != MAGIC:
            raise ValueError(f"invalid magic: {magic!r}")
        if version != VERSION:
            raise ValueError(f"unsupported version: {version}")

        if tree_count < 1:
            raise ValueError("tree file must contain at least one rank")
        trees = []
        for tree_index in range(tree_count):
            owner, particle_count, node_count = SECTION_HEADER.unpack(
                read_exact(stream, SECTION_HEADER.size, f"tree {tree_index} header")
            )
            nodes = [
                Node(*NODE_RECORD.unpack(
                    read_exact(stream, NODE_RECORD.size, f"tree {tree_index} node {node_index}")
                ))
                for node_index in range(node_count)
            ]
            trees.append((owner, particle_count, nodes))

        if stream.read(1):
            raise ValueError("unexpected trailing bytes")

    return total_particles, trees


def child_indices(nodes, parent_index):
    children = []
    seen = set()
    child = nodes[parent_index].first_child
    while child >= 0:
        if child >= len(nodes):
            raise ValueError(f"node {parent_index}: child index {child} out of range")
        if child in seen:
            raise ValueError(f"node {parent_index}: sibling cycle at node {child}")
        seen.add(child)
        children.append(child)
        child = nodes[child].next_sibling
    return children


def validate_tree(owner, particle_count, total_particles, nodes, max_depth):
    if not nodes:
        raise ValueError(f"rank {owner}: missing root")
    if owner < 0:
        raise ValueError(f"invalid owner rank {owner}")

    root = nodes[0]
    if (root.key, root.level, root.parent) != (1, 0, -1):
        raise ValueError(f"rank {owner}: invalid root topology")
    if root.next_sibling != -1:
        raise ValueError(f"rank {owner}: root has a sibling")
    if root.owner != -1 or root.is_branch or root.is_remote:
        raise ValueError(f"rank {owner}: root is not a global fill node")
    if root.particle_count != total_particles:
        raise ValueError(
            f"rank {owner}: root count {root.particle_count} != global count {total_particles}"
        )

    keys = {}
    referenced_children = set()
    local_branches = []
    for index, node in enumerate(nodes):
        if node.key in keys:
            raise ValueError(f"rank {owner}: duplicate key {node.key} at nodes {keys[node.key]} and {index}")
        keys[node.key] = index

        if not 0 <= node.level <= max_depth:
            raise ValueError(f"rank {owner}: node {index} has invalid level {node.level}")
        if node.key.bit_length() != 1 + 3 * node.level:
            raise ValueError(f"rank {owner}: node {index} key has invalid placeholder/level")
        if node.is_branch:
            if node.owner < 0:
                raise ValueError(f"rank {owner}: branch {index} has invalid owner")
            expected_remote = node.owner != owner
            if node.is_remote != expected_remote:
                raise ValueError(f"rank {owner}: branch {index} has inconsistent remote flag")
            if node.children_available == expected_remote:
                raise ValueError(f"rank {owner}: branch {index} has inconsistent child availability")
            if expected_remote:
                if node.particle_begin != -1 or node.first_child != -1:
                    raise ValueError(f"rank {owner}: remote branch {index} is not a stub")
            else:
                if node.particle_begin < 0:
                    raise ValueError(f"rank {owner}: local branch {index} has no particle range")
                local_branches.append(node)
        elif node.owner == -1:
            if node.is_remote or not node.children_available or node.particle_begin != -1:
                raise ValueError(f"rank {owner}: invalid fill node {index}")
        elif node.owner != owner or node.is_remote or not node.children_available:
            raise ValueError(f"rank {owner}: invalid local subtree node {index}")

        if node.flags & ~15 or node.child_mask & ~255:
            raise ValueError(f"rank {owner}: node {index} has unknown flags")
        if node.first_child < -1 or node.next_sibling < -1:
            raise ValueError(f"rank {owner}: node {index} has invalid link sentinel")
        if node.is_remote and node.next_sibling >= len(nodes):
            raise ValueError(f"rank {owner}: invalid remote sibling index")
        children = child_indices(nodes, index)
        if node.is_leaf and not node.is_remote:
            if children or node.child_mask:
                raise ValueError(f"rank {owner}: leaf {index} has children")
        elif not children and not node.is_remote:
            raise ValueError(f"rank {owner}: internal node {index} has no children")

        mask = 0
        child_particle_count = 0
        expected_begin = node.particle_begin
        for child_index in children:
            child = nodes[child_index]
            if child_index in referenced_children:
                raise ValueError(f"rank {owner}: child {child_index} referenced more than once")
            referenced_children.add(child_index)
            if child.parent != index:
                raise ValueError(f"rank {owner}: child {child_index} points to parent {child.parent}, expected {index}")
            if child.level != node.level + 1:
                raise ValueError(f"rank {owner}: child {child_index} has invalid level")
            if child.key >> 3 != node.key:
                raise ValueError(f"rank {owner}: child {child_index} key does not derive from parent")
            octant = child.key & 7
            mask |= 1 << octant
            if node.owner == owner:
                if child.particle_begin != expected_begin:
                    raise ValueError(f"rank {owner}: child {child_index} leaves gap/overlap in particle range")
                expected_begin += child.particle_count
            child_particle_count += child.particle_count

        if not node.is_remote and mask != node.child_mask:
            raise ValueError(f"rank {owner}: node {index} child mask mismatch")
        if children and child_particle_count != node.particle_count:
            raise ValueError(f"rank {owner}: node {index} child particle counts mismatch")

    expected_children = set(range(1, len(nodes)))
    if referenced_children != expected_children:
        missing = sorted(expected_children - referenced_children)
        raise ValueError(f"rank {owner}: unreachable nodes {missing[:8]}")

    local_branches.sort(key=lambda node: node.particle_begin)
    expected_begin = 0
    for branch in local_branches:
        if branch.particle_begin != expected_begin:
            raise ValueError(f"rank {owner}: local branch particle ranges have gap/overlap")
        expected_begin += branch.particle_count
    if expected_begin != particle_count:
        raise ValueError(
            f"rank {owner}: local branch total {expected_begin} != section count {particle_count}"
        )

    leaf_particles = sum(
        node.particle_count for node in nodes
        if node.is_leaf and node.owner == owner and not node.is_remote
    )
    if leaf_particles != particle_count:
        raise ValueError(f"rank {owner}: leaf particle total {leaf_particles} != {particle_count}")

    global_signature = tuple(sorted(
        (node.key, node.owner, node.level, node.particle_count,
         node.child_mask, node.is_branch, node.is_leaf)
        for node in nodes if node.owner == -1 or node.is_branch
    ))
    return sum(node.is_leaf for node in nodes), global_signature


def main():
    parser = argparse.ArgumentParser(description="Validate distributed hashed oct-trees stored in a .tree file")
    parser.add_argument("treefile", type=Path)
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument("--particles", type=Path, help="matching .par file for independent tree/particle checks")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input-pepc", type=Path, help="original PEPC file for conservation and coordinate/key checks")
    source.add_argument("--before-par", type=Path, help="particle dump before MPI redistribution")
    args = parser.parse_args()
    if (args.input_pepc or args.before_par) and not args.particles:
        parser.error("conservation checks require --particles")
    if not 1 <= args.max_depth <= 20:
        parser.error("--max-depth must be between 1 and 20")

    try:
        total_particles, trees = read_tree_file(args.treefile)
        owners = [owner for owner, _, _ in trees]
        if sorted(owners) != list(range(len(trees))):
            raise ValueError(f"tree owners must be ranks 0..{len(trees) - 1}, got {owners}")

        section_particles = 0
        global_signatures = []
        for owner, particle_count, nodes in trees:
            leaves, global_signature = validate_tree(
                owner, particle_count, total_particles, nodes, args.max_depth
            )
            global_signatures.append(global_signature)
            section_particles += particle_count
            print(
                f"rank={owner} particles={particle_count} nodes={len(nodes)} "
                f"leaves={leaves} hash_keys={len({node.key for node in nodes})}: OK"
            )

        if section_particles != total_particles:
            raise ValueError(
                f"section particle total {section_particles} != file total {total_particles}"
            )
        if any(signature != global_signatures[0] for signature in global_signatures[1:]):
            raise ValueError("global root/fill/branch topology differs between ranks")
        if args.particles:
            particles = read_particles(args.particles)
            check_conservation(particles, args.before_par, args.input_pepc, args.max_depth)
            check_cpu_particles(total_particles, trees, particles, args.max_depth)
            print("CPU particle intervals, ownership, remote branches and recursive reference: OK")
    except (OSError, ValueError, struct.error) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    print(f"file={args.treefile.name} trees={len(trees)} particles={total_particles}: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
