#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from pathlib import Path
import struct
import sys

from tree_visualization import save_tree_panels


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


def validate_tree(owner, particle_count, nodes, max_depth):
    if not nodes:
        raise ValueError(f"rank {owner}: missing root")
    if owner < 0:
        raise ValueError(f"invalid owner rank {owner}")

    root = nodes[0]
    if (root.key, root.level, root.parent) != (1, 0, -1):
        raise ValueError(f"rank {owner}: invalid root topology")
    if root.particle_count != particle_count:
        raise ValueError(
            f"rank {owner}: root count {root.particle_count} != section count {particle_count}"
        )

    keys = {}
    referenced_children = set()
    for index, node in enumerate(nodes):
        if node.key in keys:
            raise ValueError(f"rank {owner}: duplicate key {node.key} at nodes {keys[node.key]} and {index}")
        keys[node.key] = index

        if node.owner != owner:
            raise ValueError(f"rank {owner}: node {index} has owner {node.owner}")
        if not 0 <= node.level <= max_depth:
            raise ValueError(f"rank {owner}: node {index} has invalid level {node.level}")
        if node.key.bit_length() != 1 + 3 * node.level:
            raise ValueError(f"rank {owner}: node {index} key has invalid placeholder/level")
        if node.is_remote or node.is_branch:
            raise ValueError(f"rank {owner}: local construction marked node {index} remote/branch")
        if not node.children_available:
            raise ValueError(f"rank {owner}: local node {index} has unavailable children")

        children = child_indices(nodes, index)
        if node.is_leaf:
            if children or node.child_mask:
                raise ValueError(f"rank {owner}: leaf {index} has children")
        elif not children:
            raise ValueError(f"rank {owner}: internal node {index} has no children")

        mask = 0
        child_particle_count = 0
        expected_begin = node.particle_begin
        for child_index in children:
            child = nodes[child_index]
            referenced_children.add(child_index)
            if child.parent != index:
                raise ValueError(f"rank {owner}: child {child_index} points to parent {child.parent}, expected {index}")
            if child.level != node.level + 1:
                raise ValueError(f"rank {owner}: child {child_index} has invalid level")
            if child.key >> 3 != node.key:
                raise ValueError(f"rank {owner}: child {child_index} key does not derive from parent")
            octant = child.key & 7
            mask |= 1 << octant
            if child.particle_begin != expected_begin:
                raise ValueError(f"rank {owner}: child {child_index} leaves gap/overlap in particle range")
            expected_begin += child.particle_count
            child_particle_count += child.particle_count

        if mask != node.child_mask:
            raise ValueError(f"rank {owner}: node {index} child mask mismatch")
        if children and child_particle_count != node.particle_count:
            raise ValueError(f"rank {owner}: node {index} child particle counts mismatch")

    expected_children = set(range(1, len(nodes)))
    if referenced_children != expected_children:
        missing = sorted(expected_children - referenced_children)
        raise ValueError(f"rank {owner}: unreachable nodes {missing[:8]}")

    leaf_particles = sum(node.particle_count for node in nodes if node.is_leaf)
    if leaf_particles != particle_count:
        raise ValueError(f"rank {owner}: leaf particle total {leaf_particles} != {particle_count}")

    return sum(node.is_leaf for node in nodes)


def main():
    parser = argparse.ArgumentParser(description="Validate local hashed oct-trees stored in a .tree file")
    parser.add_argument("treefile", type=Path)
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument("--png", type=Path, help="output PNG path; default: tree filename with .png")
    args = parser.parse_args()

    try:
        total_particles, trees = read_tree_file(args.treefile)
        owners = [owner for owner, _, _ in trees]
        if sorted(owners) != list(range(len(trees))):
            raise ValueError(f"tree owners must be ranks 0..{len(trees) - 1}, got {owners}")

        section_particles = 0
        for owner, particle_count, nodes in trees:
            leaves = validate_tree(owner, particle_count, nodes, args.max_depth)
            section_particles += particle_count
            print(
                f"rank={owner} particles={particle_count} nodes={len(nodes)} "
                f"leaves={leaves} hash_keys={len({node.key for node in nodes})}: OK"
            )

        if section_particles != total_particles:
            raise ValueError(
                f"section particle total {section_particles} != file total {total_particles}"
            )

        panels = []
        for owner, particle_count, nodes in trees:
            panels.append({
                "keys": [node.key for node in nodes],
                "levels": [node.level for node in nodes],
                "parents": [node.parent for node in nodes],
                "leaves": [node.is_leaf for node in nodes],
                "particle_counts": [node.particle_count for node in nodes],
                "title": f"CPU rank {owner} | p={particle_count} n={len(nodes)}",
            })
        png_path = args.png or args.treefile.with_suffix(".png")
        save_tree_panels(panels, png_path, f"Local CPU hashed octrees — {args.treefile.name}")
    except (OSError, ValueError, struct.error) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    print(f"file={args.treefile.name} trees={len(trees)} particles={total_particles}: OK")
    print(f"image={png_path}: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
