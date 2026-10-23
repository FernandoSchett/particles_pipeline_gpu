"""Small host-only oracles for particle conservation and distributed octrees.

These routines use binary search, recursive interval subdivision, and bit
interleaving. They do not call production tree builders or GPU kernels.
"""
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
import struct

PAR_RECORD = struct.Struct('<iqddd')


@dataclass(frozen=True)
class Particle:
    rank: int
    key: int
    coords: tuple


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_particles(path):
    raw = Path(path).read_bytes()
    require(len(raw) >= 8, 'missing particle header')
    count, = struct.unpack_from('<q', raw)
    require(count >= 0 and len(raw) == 8 + count * PAR_RECORD.size,
            'particle file size does not match its count')
    return [Particle(r, k, (x, y, z)) for r, k, x, y, z in PAR_RECORD.iter_unpack(raw[8:])]


def read_pepc_coordinates(path):
    """Read the 128-byte header / 216-byte particle layout used by this project."""
    raw = Path(path).read_bytes()
    require(len(raw) >= 128, 'truncated PEPC header')
    count, = struct.unpack_from('<q', raw)
    require(count >= 0 and len(raw) == 128 + count * 216,
            'unsupported or truncated PEPC layout (expected 216-byte records)')
    return [struct.unpack_from('<ddd', raw, 128 + i * 216) for i in range(count)]


def coordinate_multiset(coords):
    require(all(all(math.isfinite(v) for v in p) for p in coords),
            'nonfinite particle coordinates')
    # Preserve multiplicities, including particles with coincident coordinates.
    return Counter(tuple(p) for p in coords)


def domain(coords):
    coordinate_multiset(coords)
    if not coords:
        return (0., 0., 0.), 1.
    origin = tuple(min(p[a] for p in coords) for a in range(3))
    length = max(max(p[a] for p in coords) - origin[a] for a in range(3))
    side = math.nextafter(length, math.inf) if length else 1.
    require(math.isfinite(side), 'nonfinite coordinate span')
    return origin, side


def morton_key(coords, origin, side, depth):
    cells = []
    for value, minimum in zip(coords, origin):
        offset = value - minimum
        require(0 <= offset <= side, 'coordinate outside Morton domain')
        cells.append(min((1 << depth) - 1, int(offset / side * (1 << depth))))
    return sum(((cells[axis] >> bit) & 1) << (3 * bit + axis)
               for bit in range(depth) for axis in range(3))


def check_conservation(particles, before=None, input_pepc=None, depth=15):
    """Compare multisets, never just totals; PEPC also provides a key oracle."""
    actual = coordinate_multiset([p.coords for p in particles])
    if before is not None:
        previous = read_particles(before)
        require(actual == coordinate_multiset([p.coords for p in previous]),
                'particle loss/duplication or coordinate changes during redistribution')
        require(Counter((p.key, p.coords) for p in particles) ==
                Counter((p.key, p.coords) for p in previous),
                'particle keys changed during redistribution')
    if input_pepc is not None:
        coords = read_pepc_coordinates(input_pepc)
        require(actual == coordinate_multiset(coords),
                'particle loss/duplication relative to PEPC input')
        origin, side = domain(coords)
        require(all(p.key == morton_key(p.coords, origin, side, depth) for p in particles),
                'Morton keys disagree with independent coordinate reference')


def partition(particles, ranks, depth):
    require(ranks > 0 and 1 <= depth <= 20, 'invalid rank count or depth')
    limit = 1 << (3 * depth)
    keys = sorted(p.key for p in particles)
    require(all(0 <= k < limit for k in keys), 'particle key outside Morton space')
    splitters = [keys[(len(keys) * r + ranks - 1) // ranks - 1]
                 for r in range(1, ranks)] if keys else [0] * (ranks - 1)
    local = [[] for _ in range(ranks)]
    for p in particles:
        require(0 <= p.rank < ranks, 'invalid particle owner')
        require(p.rank == bisect_left(splitters, p.key),
                'particle assigned to wrong Morton partition')
        local[p.rank].append(p.key)
    for values in local:
        require(values == sorted(values), 'local particle keys are not sorted')
    boundaries = [0] + [s + 1 for s in splitters] + [limit]
    return keys, local, boundaries


def node_interval(key, depth):
    level = (key.bit_length() - 1) // 3
    width = 1 << (3 * (depth - level))
    begin = (key - (1 << (3 * level))) * width
    return begin, begin + width


def count_interval(keys, begin, end):
    return bisect_left(keys, end) - bisect_left(keys, begin)


def cornerstone_reference(keys, depth, ncrit):
    """Build the complete adaptive octree, including empty octants."""
    require(ncrit > 0, 'invalid occupancy threshold')
    keys = sorted(keys)
    leaves, nodes = [], {}

    def visit(key, level, begin, end):
        count = count_interval(keys, begin, end)
        split = count > ncrit and level < depth
        nodes[key] = split
        if split:
            width = (end - begin) // 8
            for octant in range(8):
                visit(key * 8 + octant, level + 1,
                      begin + octant * width, begin + (octant + 1) * width)
        else:
            leaves.append((begin, end, count))
    visit(1, 0, 0, 1 << (3 * depth))
    return leaves, nodes


def hashed_reference(local, boundaries, depth):
    """Build occupied cells per rank, then assemble local details and remote stubs."""
    details, branches = [], {}
    for owner, keys in enumerate(local):
        nodes = {}
        low, high = boundaries[owner:owner + 2]

        def visit(key, level, begin, end, in_branch=False):
            first, last = bisect_left(keys, begin), bisect_left(keys, end)
            if first == last:
                return
            inside = low <= begin and end <= high
            branch = level > 0 and inside and not in_branch
            leaf = level == depth or (level > 0 and last - first == 1 and inside)
            if in_branch or branch:
                nodes[key] = dict(owner=owner, level=level, count=last-first,
                                  begin=first, leaf=leaf, branch=branch,
                                  remote=False, available=True, mask=0)
            if branch:
                branches[key] = owner
            if not leaf:
                width = (end-begin)//8
                for octant in range(8):
                    visit(key*8+octant, level+1, begin+octant*width,
                          begin+(octant+1)*width, in_branch or branch)
            if key in nodes and not leaf:
                nodes[key]['mask'] = sum(1 << o for o in range(8) if key*8+o in nodes)
        visit(1, 0, 0, 1 << (3*depth))
        details.append(nodes)

    global_keys = sorted(k for keys in local for k in keys)
    fills = {1}
    for key in branches:
        ancestor = key >> 3
        while ancestor:
            fills.add(ancestor)
            ancestor >>= 3
    result = []
    for rank in range(len(local)):
        nodes = {k: dict(v) for k, v in details[rank].items()}
        for key, owner in branches.items():
            if owner != rank:
                nodes[key] = dict(details[owner][key], begin=-1, remote=True, available=False)
        for key in fills:
            begin, end = node_interval(key, depth)
            nodes[key] = dict(owner=-1, level=(key.bit_length()-1)//3,
                              count=count_interval(global_keys, begin, end), begin=-1,
                              leaf=False, branch=False, remote=False, available=True,
                              mask=0)
        for key in fills:
            nodes[key]['mask'] = sum(1 << o for o in range(8) if key*8+o in nodes)
            nodes[key]['leaf'] = not nodes[key]['mask']
        result.append(nodes)
    return result


def check_cpu_particles(total, trees, particles, depth):
    require(total == len(particles), 'CPU tree total differs from particle file')
    keys, local, boundaries = partition(particles, len(trees), depth)
    expected = hashed_reference(local, boundaries, depth)
    for owner, count, nodes in trees:
        require(count == len(local[owner]), f'rank {owner}: section particle count mismatch')
        require({n.key for n in nodes} == set(expected[owner]),
                f'rank {owner}: topology differs from recursive hashed reference')
        for node in nodes:
            ref = expected[owner][node.key]
            actual = dict(owner=node.owner, level=node.level, count=node.particle_count,
                          begin=node.particle_begin, leaf=node.is_leaf, branch=node.is_branch,
                          remote=node.is_remote, available=node.children_available, mask=node.child_mask)
            require(actual == ref, f'rank {owner}: node {node.key} differs from particle/reference metadata')
            begin, end = node_interval(node.key, depth)
            require(node.particle_count == count_interval(keys, begin, end),
                    f'rank {owner}: node {node.key} Morton interval count mismatch')
            if node.owner == owner:
                first, last = bisect_left(local[owner], begin), bisect_left(local[owner], end)
                require(node.particle_begin == first and node.particle_count == last-first,
                        f'rank {owner}: node {node.key} particle range mismatch')
    # Remote references must match an actual branch in the advertised owner's tree.
    by_rank = {owner: {n.key: n for n in nodes} for owner, _, nodes in trees}
    for rank, _, nodes in trees:
        for node in nodes:
            if node.is_remote:
                target = by_rank.get(node.owner, {}).get(node.key)
                require(target is not None and target.is_branch and not target.is_remote,
                        f'rank {rank}: unresolved remote branch {node.key}')
                require((node.particle_count, node.child_mask, node.is_leaf) ==
                        (target.particle_count, target.child_mask, target.is_leaf),
                        f'rank {rank}: remote summary differs from owner')


def check_gpu_particles(tree, particles):
    depth = tree['max_depth']
    require(tree['particles'] == len(particles), 'GPU tree total differs from particle file')
    keys, _, _ = partition(particles, tree['rank_count'], depth)
    leaves, nodes = cornerstone_reference(keys, depth, tree['ncrit'])
    require(tree['cornerstone'] == [leaf[0] for leaf in leaves] + [1 << (3*depth)],
            'cornerstone differs from recursive reference')
    require(tree['leaf_counts'] == [leaf[2] for leaf in leaves],
            'GPU leaf counts disagree with actual Morton intervals')
    require(tree['node_keys'] == sorted(nodes), 'GPU topology differs from recursive reference')
    for key, child in zip(tree['node_keys'], tree['child_offsets']):
        require(bool(child) == nodes[key], f'GPU node {key}: incorrect leaf/internal state')
