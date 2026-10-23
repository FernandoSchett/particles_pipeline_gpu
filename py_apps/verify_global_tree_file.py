#!/usr/bin/env python3
import argparse
from pathlib import Path
import struct
import sys
from tree_reference import read_particles, check_conservation, check_gpu_particles

MAGIC = b"PSFCGTRE"
VERSION = 2
HEADER = struct.Struct("<8sIIIIQQQ")


def read_values(stream, fmt, count, description):
    size = fmt.size * count
    data = stream.read(size)
    if len(data) != size:
        raise ValueError(f"truncated {description}: expected {size} bytes, got {len(data)}")
    return [value[0] for value in struct.iter_unpack(fmt.format, data)]


def node_level(key):
    return (key.bit_length() - 1) // 3


def validate(path):
    with path.open("rb") as stream:
        header = stream.read(HEADER.size)
        if len(header) != HEADER.size:
            raise ValueError("truncated header")
        magic, version, max_depth, ncrit, rank_count, particles, leaves, nodes = HEADER.unpack(header)
        if magic != MAGIC:
            raise ValueError(f"invalid magic: {magic!r}")
        if version != VERSION:
            raise ValueError(f"unsupported version: {version}")
        if not 1 <= max_depth <= 20 or ncrit < 1 or rank_count < 1 or leaves < 1 or nodes < 1:
            raise ValueError("invalid tree dimensions")

        cornerstone = read_values(stream, struct.Struct("<Q"), leaves + 1, "cornerstone")
        leaf_counts = read_values(stream, struct.Struct("<Q"), leaves, "leaf counts")
        node_keys = read_values(stream, struct.Struct("<Q"), nodes, "node keys")
        child_offsets = read_values(stream, struct.Struct("<i"), nodes, "child offsets")
        level_offsets = read_values(stream, struct.Struct("<i"), max_depth + 2, "level offsets")
        if stream.read(1):
            raise ValueError("unexpected trailing bytes")

    key_space_end = 1 << (3 * max_depth)
    if cornerstone[0] != 0 or cornerstone[-1] != key_space_end:
        raise ValueError("cornerstone does not cover complete Morton key space")
    if any(left >= right for left, right in zip(cornerstone, cornerstone[1:])):
        raise ValueError("cornerstone boundaries are not strictly increasing")
    for begin, end in zip(cornerstone, cornerstone[1:]):
        width = end - begin
        if width & (width - 1) or (width.bit_length() - 1) % 3:
            raise ValueError(f"invalid octree leaf interval [{begin}, {end})")
        if begin % width:
            raise ValueError(f"unaligned octree leaf interval [{begin}, {end})")
    if sum(leaf_counts) != particles:
        raise ValueError(f"leaf particle total {sum(leaf_counts)} != header {particles}")
    if any(count > ncrit and end - begin > 1 for count, begin, end in
           zip(leaf_counts, cornerstone, cornerstone[1:])):
        raise ValueError("splittable leaf exceeds ncrit")

    if node_keys[0] != 1 or node_keys != sorted(set(node_keys)):
        raise ValueError("node keys are not unique breadth-first Morton order")
    if level_offsets[0] != 0 or level_offsets[-1] != nodes:
        raise ValueError("invalid level offset endpoints")
    if any(left > right for left, right in zip(level_offsets, level_offsets[1:])):
        raise ValueError("level offsets are not monotonic")

    key_to_index = {key: index for index, key in enumerate(node_keys)}
    for level in range(max_depth + 1):
        first, last = level_offsets[level], level_offsets[level + 1]
        if any(node_level(key) != level for key in node_keys[first:last]):
            raise ValueError(f"level {level} contains key with wrong placeholder level")

    for key in node_keys:
        if key != 1 and key >> 3 not in key_to_index:
            raise ValueError(f"orphan GPU node {key}")
    internal_count = 0
    for index, (key, child) in enumerate(zip(node_keys, child_offsets)):
        if child == 0:
            continue
        internal_count += 1
        if child < 0 or child + 8 > nodes:
            raise ValueError(f"node {index}: invalid first child {child}")
        expected = list(range(key << 3, (key << 3) + 8))
        if node_keys[child:child + 8] != expected:
            raise ValueError(f"node {index}: children are not complete Morton octants")
        if any(key_to_index.get(child_key) != child + offset
               for offset, child_key in enumerate(expected)):
            raise ValueError(f"node {index}: child lookup mismatch")

    leaf_keys = set()
    for begin, end in zip(cornerstone, cornerstone[1:]):
        level = max_depth - ((end - begin).bit_length() - 1) // 3
        leaf_keys.add((1 << (3 * level)) | (begin >> (3 * (max_depth - level))))
    topology_leaves = {key for key, child in zip(node_keys, child_offsets) if child == 0}
    if leaf_keys != topology_leaves:
        raise ValueError("cornerstone leaves and linked-tree leaves differ")

    return {
        "particles": particles,
        "leaf_count": leaves,
        "node_count": nodes,
        "internal_count": internal_count,
        "max_depth": max_depth,
        "ncrit": ncrit,
        "rank_count": rank_count,
        "node_keys": node_keys,
        "child_offsets": child_offsets,
        "leaf_counts": leaf_counts,
        "cornerstone": cornerstone,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate distributed GPU octree .gtree file")
    parser.add_argument("treefile", type=Path)
    parser.add_argument("--particles", type=Path, help="matching .par file for independent reference checks")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input-pepc", type=Path, help="original PEPC file for conservation and coordinate/key checks")
    source.add_argument("--before-par", type=Path, help="particle dump before MPI redistribution")
    args = parser.parse_args()
    if (args.input_pepc or args.before_par) and not args.particles:
        parser.error("conservation checks require --particles")
    try:
        tree = validate(args.treefile)
        if args.particles:
            particles = read_particles(args.particles)
            check_conservation(particles, args.before_par, args.input_pepc, tree['max_depth'])
            check_gpu_particles(tree, particles)
            print("GPU particle intervals, partition ownership and recursive reference: OK")
    except (OSError, ValueError, struct.error) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(
        f"file={args.treefile.name} ranks={tree['rank_count']} particles={tree['particles']} "
        f"leaves={tree['leaf_count']} nodes={tree['node_count']} "
        f"internal={tree['internal_count']} depth={tree['max_depth']} ncrit={tree['ncrit']}: OK"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
