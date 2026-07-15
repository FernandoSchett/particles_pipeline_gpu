#!/usr/bin/env python3
import math
import os
from pathlib import Path
import tempfile

_matplotlib_config = Path(tempfile.gettempdir()) / f"p_sfc_matplotlib_{os.getuid()}"
_matplotlib_config.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_config))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def key_position(key, level):
    prefix = key - (1 << (3 * level))
    return (prefix + 0.5) / (1 << (3 * level))


def draw_tree(axis, keys, levels, parents, leaves, particle_counts, title):
    positions = [key_position(key, level) for key, level in zip(keys, levels)]
    segments = []
    edge_level_limit = max(levels, default=0) if len(keys) <= 10000 else 6
    for index, parent in enumerate(parents):
        if parent >= 0 and levels[index] <= edge_level_limit:
            segments.append([
                (positions[parent], -levels[parent]),
                (positions[index], -levels[index]),
            ])
    if segments:
        axis.add_collection(LineCollection(segments, colors="#9aa4b2", linewidths=0.35,
                                           alpha=0.55, rasterized=True))

    internal_x = [positions[index] for index, leaf in enumerate(leaves) if not leaf]
    internal_y = [-levels[index] for index, leaf in enumerate(leaves) if not leaf]
    leaf_x = [positions[index] for index, leaf in enumerate(leaves) if leaf]
    leaf_y = [-levels[index] for index, leaf in enumerate(leaves) if leaf]
    leaf_sizes = [max(2.0, min(18.0, 2.0 + math.log2(particle_counts[index] + 1)))
                  for index, leaf in enumerate(leaves) if leaf]

    axis.scatter(internal_x, internal_y, s=2.0, color="#2563eb", alpha=0.7,
                 linewidths=0, rasterized=True)
    axis.scatter(leaf_x, leaf_y, s=leaf_sizes, color="#f97316", alpha=0.85,
                 linewidths=0, rasterized=True)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(-max(levels, default=0) - 0.7, 0.7)
    axis.set_title(title, fontsize=8)
    axis.set_xlabel("Morton/Z", fontsize=6)
    axis.set_ylabel("nível", fontsize=6)
    axis.tick_params(labelsize=5, length=2)
    axis.grid(axis="y", linewidth=0.25, alpha=0.35)


def save_tree_panels(panels, output, heading):
    panel_count = len(panels)
    columns = max(1, math.ceil(math.sqrt(panel_count)))
    rows = math.ceil(panel_count / columns)
    figure, axes = plt.subplots(
        rows, columns, squeeze=False,
        figsize=(max(4.5, columns * 2.2), max(3.5, rows * 2.0)),
        constrained_layout=True,
    )
    flat_axes = list(axes.flat)
    for axis, panel in zip(flat_axes, panels):
        draw_tree(axis, **panel)
    for axis in flat_axes[panel_count:]:
        axis.set_visible(False)
    figure.suptitle(heading, fontsize=11)
    figure.savefig(output, dpi=140)
    plt.close(figure)
