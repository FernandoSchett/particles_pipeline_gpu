"""Positive and mutation tests for the independent correctness oracles."""
from dataclasses import replace
from pathlib import Path
import struct
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tree_reference import (Particle, check_conservation, check_cpu_particles,
                            check_gpu_particles, cornerstone_reference, hashed_reference,
                            partition, morton_key, read_particles)
from verify_tree_file import Node, validate_tree, read_tree_file
from verify_global_tree_file import validate as validate_gpu


def cpu_fixture():
    # At depth two, splitter 7 makes the first octant a complete local branch.
    particles = [Particle(r, k, (float(k), 0., 0.)) for r, keys in enumerate(((0, 7), (32, 33))) for k in keys]
    # Hand-written topology, independent of hashed_reference().
    trees = []
    for rank in range(2):
        local = 8 if rank == 0 else 12
        nodes = [Node(1, -1, 0, -1, 1, -1, 17, -1, 4, 2)]
        # Keep children in Morton order, with local descendants appended.
        for key in (8, 12):
            own = 0 if key == 8 else 1
            nodes.append(Node(key, own, 1, 0, 3 if own == rank else -1,
                              2 if key == 8 else -1, 129 if key == 8 else 3, 0 if own == rank else -1,
                              2, 6 if own == rank else 12))
        parent = 1 if rank == 0 else 2
        nodes += [Node(local*8, rank, 2, parent, -1, 4, 0, 0, 1, 3),
                  Node(local*8+(7 if rank == 0 else 1), rank, 2, parent, -1, -1, 0, 1, 1, 3)]
        trees.append((rank, 2, nodes))
    return particles, trees


def gpu_fixture():
    # Root splits once: first and last octants contain two particles each.
    particles = [Particle(r, k, (float(k), 0., 0.)) for r, keys in enumerate(((0, 1), (62, 63))) for k in keys]
    tree = dict(particles=4, rank_count=2, max_depth=2, ncrit=2,
                cornerstone=list(range(0, 65, 8)), leaf_counts=[2, 0, 0, 0, 0, 0, 0, 2],
                node_keys=[1]+list(range(8, 16)), child_offsets=[1]+[0]*8)
    return particles, tree


class CorrectnessTests(unittest.TestCase):
    def test_known_morton_boundaries(self):
        for octant in range(8):
            xyz = tuple(.5 if octant & (1 << a) else 0. for a in range(3))
            self.assertEqual(morton_key(xyz, (0, 0, 0), 1., 2), octant << 3)
        self.assertEqual(morton_key((1, 1, 1), (0, 0, 0), 1., 2), 63)

    def test_cpu_handwritten_tree(self):
        p, trees = cpu_fixture()
        for rank, count, nodes in trees:
            validate_tree(rank, count, 4, nodes, 2)
        check_cpu_particles(4, trees, p, 2)

    def test_cpu_mutations(self):
        for field, value in [('particle_begin', 1), ('particle_count', 2),
                             ('key', 66), ('owner', 1), ('flags', 2)]:
            p, trees = cpu_fixture()
            trees[0][2][3] = replace(trees[0][2][3], **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_cpu_particles(4, trees, p, 2)

    def test_remote_branch_mutations(self):
        for field, value in [('owner', 9), ('particle_count', 3),
                             ('child_mask', 7), ('flags', 4), ('particle_begin', 0)]:
            p, trees = cpu_fixture()
            trees[0][2][2] = replace(trees[0][2][2], **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_cpu_particles(4, trees, p, 2)

    def test_cycle_rejected(self):
        _, trees = cpu_fixture()
        trees[0][2][1] = replace(trees[0][2][1], next_sibling=1)
        with self.assertRaises(ValueError):
            validate_tree(0, 2, 4, trees[0][2], 2)

    def test_gpu_handwritten_tree(self):
        p, tree = gpu_fixture()
        check_gpu_particles(tree, p)

    def test_gpu_count_mutation_preserving_total(self):
        p, tree = gpu_fixture()
        tree['leaf_counts'][0], tree['leaf_counts'][1] = 1, 1
        with self.assertRaises(ValueError):
            check_gpu_particles(tree, p)

    def test_gpu_wrong_refinement_and_node(self):
        p, tree = gpu_fixture()
        for field, value in [('cornerstone', [0, 64]), ('node_keys', [1, 8]),
                             ('child_offsets', [0]*9)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_gpu_particles(dict(tree, **{field: value}), p)

    def test_repeated_keys_and_empty_ranks(self):
        p = [Particle(0, 7, (0., 0., 0.)) for _ in range(100)]
        keys, local, boundaries = partition(p, 4, 2)
        self.assertEqual(list(map(len, local)), [100, 0, 0, 0])
        leaves, _ = cornerstone_reference(keys, 2, 4)
        self.assertIn((7, 8, 100), leaves)  # Occupancy may exceed ncrit only at maximum depth.
        expected = hashed_reference(local, boundaries, 2)
        self.assertTrue(any(n['remote'] for n in expected[1].values()))
        bad = p[:-1]+[replace(p[-1], rank=1)]
        with self.assertRaises(ValueError):
            partition(bad, 4, 2)

    def test_concentrated_and_empty_reference(self):
        leaves, nodes = cornerstone_reference([0]*65, 3, 64)
        self.assertIn((0, 1, 65), leaves)
        self.assertEqual(sum(x[2] for x in leaves), 65)
        self.assertEqual(cornerstone_reference([], 3, 64), ([(0, 512, 0)], {1: False}))

    def test_multiset_detects_loss_and_duplicate_with_same_total(self):
        p, _ = gpu_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'before.par'
            path.write_bytes(struct.pack('<q', len(p)) + b''.join(
                struct.pack('<iqddd', x.rank, x.key, *x.coords) for x in p))
            check_conservation(list(reversed(p)), before=path)
            with self.assertRaises(ValueError):
                check_conservation(p[:-1]+[p[0]], before=path)
            with self.assertRaises(ValueError):
                check_conservation([replace(p[0], key=2)]+p[1:], before=path)
            with path.open('ab') as stream:
                stream.write(b'garbage')
            with self.assertRaises(ValueError):
                read_particles(path)

    def test_pepc_conservation_and_coordinate_key_oracle(self):
        from run_tree_correctness import write_pepc
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'input.mpi'
            write_pepc(path, [(-1., 0., 0.), (1., 0., 0.)])
            # With the padded side, the maximum is in the last x cell.
            particles = [Particle(0, 0, (-1., 0., 0.)), Particle(0, 9, (1., 0., 0.))]
            check_conservation(particles, input_pepc=path, depth=2)
            with self.assertRaises(ValueError):
                check_conservation([particles[0], particles[0]], input_pepc=path, depth=2)
            with self.assertRaises(ValueError):
                check_conservation([particles[0], replace(particles[1], key=0)], input_pepc=path, depth=2)

    def test_gpu_binary_and_particle_crosscheck(self):
        particles, tree = gpu_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'tree.gtree'
            def write(counts, keys):
                data = struct.pack('<8sIIIIQQQ', b'PSFCGTRE', 2, 2, 2, 2, 4, 8, 9)
                for values, code in [(tree['cornerstone'], 'Q'), (counts, 'Q'),
                                     (keys, 'Q'), (tree['child_offsets'], 'i'),
                                     ([0, 1, 9, 9], 'i')]:
                    data += struct.pack('<' + code*len(values), *values)
                path.write_bytes(data)
            write(tree['leaf_counts'], tree['node_keys'])
            check_gpu_particles(validate_gpu(path), particles)
            # Still structurally valid with the same total, but spatially wrong.
            write([1, 1, 0, 0, 0, 0, 0, 2], tree['node_keys'])
            structurally_valid = validate_gpu(path)
            with self.assertRaises(ValueError):
                check_gpu_particles(structurally_valid, particles)
            write(tree['leaf_counts'], [1, 8, 9, 10, 11, 12, 13, 14, 64])
            with self.assertRaises(ValueError):
                validate_gpu(path)

    def test_binary_readers_reject_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad'
            path.write_bytes(b'PSFC')
            for reader in (read_tree_file, validate_gpu):
                with self.subTest(reader=reader.__name__), self.assertRaises(ValueError):
                    reader(path)


if __name__ == '__main__':
    unittest.main()
