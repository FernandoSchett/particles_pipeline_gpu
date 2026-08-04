#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import struct
import sys

import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np


PAR_HEADER = struct.Struct("<q")
PARTICLE_RECORD = np.dtype([
    ("rank", "<i4"),
    ("key", "<i8"),
    ("coord", "<f8", (3,)),
])
TREE_MAGIC = b"PSFCTREE"
TREE_VERSION = 1
TREE_HEADER = struct.Struct("<8sIIQ")
TREE_SECTION = struct.Struct("<iQQ")
TREE_NODE = struct.Struct("<QiiiiiBiQB")


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


def read_particles(path):
    with path.open("rb") as stream:
        particle_count = PAR_HEADER.unpack(
            read_exact(stream, PAR_HEADER.size, "particle header")
        )[0]
        if particle_count < 0:
            raise ValueError(f"negative particle count {particle_count}")
        particles = np.fromfile(stream, dtype=PARTICLE_RECORD, count=particle_count)
        if len(particles) != particle_count:
            raise ValueError(
                f"truncated particles: expected {particle_count}, got {len(particles)}"
            )
        if stream.read(1):
            raise ValueError("unexpected trailing bytes in particle file")
    return particles


def read_trees(path):
    with path.open("rb") as stream:
        magic, version, tree_count, total_particles = TREE_HEADER.unpack(
            read_exact(stream, TREE_HEADER.size, "tree header")
        )
        if magic != TREE_MAGIC:
            raise ValueError(f"invalid tree magic {magic!r}")
        if version != TREE_VERSION:
            raise ValueError(f"unsupported tree version {version}")

        trees = []
        for tree_index in range(tree_count):
            rank, local_particle_count, node_count = TREE_SECTION.unpack(
                read_exact(stream, TREE_SECTION.size, f"tree {tree_index} header")
            )
            nodes = [
                Node(*TREE_NODE.unpack(
                    read_exact(stream, TREE_NODE.size,
                               f"tree {tree_index} node {node_index}")
                ))
                for node_index in range(node_count)
            ]
            trees.append((rank, local_particle_count, nodes))
        if stream.read(1):
            raise ValueError("unexpected trailing bytes in tree file")
    return total_particles, trees


def infer_box_size(particles):
    if not len(particles):
        return 1.0
    maximum = float(np.max(particles["coord"]))
    if maximum <= 0:
        return 1.0
    return 10.0 ** math.ceil(math.log10(maximum))


def node_center(node, box_size, max_depth):
    if node.level < 0 or node.level > max_depth:
        raise ValueError(f"node key {node.key}: invalid level {node.level}")
    marker = 1 << (3 * node.level)
    if node.key < marker or node.key >= 2 * marker:
        raise ValueError(f"node key {node.key}: invalid placeholder bit")

    path = node.key - marker
    origin = np.zeros(3, dtype=float)
    cell_size = float(box_size)
    for depth in range(node.level):
        shift = 3 * (node.level - depth - 1)
        octant = (path >> shift) & 7
        cell_size *= 0.5
        origin[0] += cell_size if octant & 1 else 0.0
        origin[1] += cell_size if octant & 2 else 0.0
        origin[2] += cell_size if octant & 4 else 0.0
    return origin + 0.5 * cell_size


def add_segments(axis, segments, color, width=1.0, alpha=0.75):
    collection = Line3DCollection(segments, colors=color, linewidths=width, alpha=alpha)
    axis.add_collection3d(collection)
    return collection


def add_nodes(axis, centers, indices, color, marker, size, label):
    points = centers[indices] if indices else np.empty((0, 3))
    return axis.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        color=color, marker=marker, s=size, depthshade=False, label=label,
    )


def node_category(node, rank, index):
    if index == 0:
        return "root"
    if node.is_branch and node.is_remote:
        return "remote_branches"
    if node.is_branch:
        return "local_branches"
    if node.owner == -1:
        return "fill_nodes"
    return "local_subtree"


def create_rank_figure(rank, local_particle_count, nodes, particles,
                       box_size, max_depth, show_leaf_links):
    figure = plt.figure(figsize=(12, 8))
    axis = figure.add_axes([0.04, 0.08, 0.70, 0.86], projection="3d")
    axis.set_title(f"Rank {rank} — árvore materializada ({len(nodes)} nós)")
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    axis.set_xlim(0, box_size)
    axis.set_ylim(0, box_size)
    axis.set_zlim(0, box_size)
    axis.set_box_aspect((1, 1, 1))

    local_mask = particles["rank"] == rank
    remote_mask = ~local_mask
    local_coords = particles["coord"][local_mask]
    remote_coords = particles["coord"][remote_mask]
    local_particles = axis.scatter(
        local_coords[:, 0], local_coords[:, 1], local_coords[:, 2],
        color=plt.get_cmap("tab10")(rank % 10), s=14, alpha=0.9,
        depthshade=False, label="partículas locais",
    )
    remote_particles = axis.scatter(
        remote_coords[:, 0], remote_coords[:, 1], remote_coords[:, 2],
        color="#808080", s=7, alpha=0.18,
        depthshade=False, label="partículas remotas",
    )

    centers = np.asarray([
        node_center(node, box_size, max_depth) for node in nodes
    ])
    categories = {
        "root": [],
        "fill_nodes": [],
        "local_branches": [],
        "remote_branches": [],
        "local_subtree": [],
    }
    edge_segments = {category: [] for category in categories}
    for index, node in enumerate(nodes):
        category = node_category(node, rank, index)
        categories[category].append(index)
        if node.parent >= 0:
            if node.parent >= len(nodes):
                raise ValueError(f"rank {rank}: node {index} has invalid parent {node.parent}")
            edge_segments[category].append([centers[node.parent], centers[index]])

    root_artist = add_nodes(axis, centers, categories["root"],
                            "#111111", "*", 110, "root global")
    fill_artist = add_nodes(axis, centers, categories["fill_nodes"],
                            "#06b6d4", "o", 18, "fill nodes")
    local_branch_artist = add_nodes(axis, centers, categories["local_branches"],
                                    "#dc2626", "D", 34, "branches locais")
    remote_branch_artist = add_nodes(axis, centers, categories["remote_branches"],
                                     "#9333ea", "X", 38, "branches remotas")

    local_internal = [
        index for index in categories["local_subtree"] if not nodes[index].is_leaf
    ]
    local_leaves = [
        index for index in categories["local_subtree"] if nodes[index].is_leaf
    ]
    internal_artist = add_nodes(axis, centers, local_internal,
                                "#2563eb", "o", 13, "nós internos locais")
    leaf_artist = add_nodes(axis, centers, local_leaves,
                            "#16a34a", "^", 24, "folhas locais")

    fill_edges = add_segments(axis, edge_segments["fill_nodes"], "#67e8f9", 0.9, 0.7)
    local_branch_edges = add_segments(
        axis, edge_segments["local_branches"], "#ef4444", 1.5, 0.9
    )
    remote_branch_edges = add_segments(
        axis, edge_segments["remote_branches"], "#a855f7", 1.3, 0.9
    )
    subtree_edges = add_segments(
        axis, edge_segments["local_subtree"], "#3b82f6", 0.75, 0.6
    )

    rank_particles = particles[local_mask]
    if len(rank_particles) != local_particle_count:
        raise ValueError(
            f"rank {rank}: .par has {len(rank_particles)} local particles, "
            f".tree reports {local_particle_count}"
        )
    leaf_segments = []
    local_branch_leaves = [
        index for index in categories["local_branches"] if nodes[index].is_leaf
    ]
    for index in local_leaves + local_branch_leaves:
        node = nodes[index]
        if node.is_remote or not node.children_available or node.particle_begin < 0:
            continue
        end = node.particle_begin + node.particle_count
        if end > len(rank_particles):
            raise ValueError(f"rank {rank}: leaf {index} particle interval out of range")
        for coordinate in rank_particles["coord"][node.particle_begin:end]:
            leaf_segments.append([centers[index], coordinate])
    leaf_link_artist = add_segments(axis, leaf_segments, "#22c55e", 0.45, 0.35)
    leaf_link_artist.set_visible(show_leaf_links)

    controls = {
        "partículas locais": [local_particles],
        "partículas remotas": [remote_particles],
        "fill nodes": [fill_artist, fill_edges],
        "branches locais": [local_branch_artist, local_branch_edges],
        "branches remotas": [remote_branch_artist, remote_branch_edges],
        "subárvore local": [internal_artist, leaf_artist, subtree_edges],
        "folha → partículas": [leaf_link_artist],
    }
    control_axis = figure.add_axes([0.77, 0.28, 0.21, 0.38])
    labels = list(controls)
    initial = [all(artist.get_visible() for artist in controls[label]) for label in labels]
    check_buttons = CheckButtons(control_axis, labels, initial)

    def toggle(label):
        artists = controls[label]
        visible = not all(artist.get_visible() for artist in artists)
        for artist in artists:
            artist.set_visible(visible)
        figure.canvas.draw_idle()

    check_buttons.on_clicked(toggle)
    figure._tree_check_buttons = check_buttons
    figure._tree_root_artist = root_artist
    return figure


def main():
    parser = argparse.ArgumentParser(
        description="Visualização 3D interativa de partículas e árvores distribuídas CPU"
    )
    parser.add_argument("par_file", type=Path, help="arquivo .par com todas as partículas")
    parser.add_argument("tree_file", type=Path, help="arquivo .tree distribuído")
    parser.add_argument("--box-size", type=float,
                        help="tamanho da caixa; inferido das coordenadas quando omitido")
    parser.add_argument("--max-depth", type=int, default=15,
                        help="profundidade da Morton key usada para validar nós")
    parser.add_argument("--show-leaf-links", action="store_true",
                        help="mostrar inicialmente ligações folha-partícula")
    args = parser.parse_args()

    try:
        particles = read_particles(args.par_file)
        total_particles, trees = read_trees(args.tree_file)
        if len(particles) != total_particles:
            raise ValueError(
                f".par has {len(particles)} particles, .tree reports {total_particles}"
            )
        ranks = [rank for rank, _, _ in trees]
        if sorted(ranks) != list(range(len(trees))):
            raise ValueError(f"tree ranks must be 0..{len(trees) - 1}, got {ranks}")
        box_size = args.box_size or infer_box_size(particles)
        if box_size <= 0:
            raise ValueError("box size must be positive")

        for rank, local_particle_count, nodes in trees:
            create_rank_figure(
                rank, local_particle_count, nodes, particles,
                box_size, args.max_depth, args.show_leaf_links,
            )
        print(
            f"particles={len(particles)} ranks={len(trees)} box_size={box_size:g}; "
            "close all figure windows to exit"
        )
        plt.show()
    except (OSError, ValueError, struct.error) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
