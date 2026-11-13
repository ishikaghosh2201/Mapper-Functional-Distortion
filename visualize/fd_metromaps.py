"""Draw continuous maps between the original Mapper graphs."""

from collections import defaultdict
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch

from cereeberus.draw.draw import _format_node_label

__all__ = ["draw_fd_metromaps", "layout_from_mapper", "default_layout"]

INK = "#222222"


def layout_from_mapper(mapper):
    """Return the positions used by ceREEBerus's MapperGraph.draw."""
    return {v: np.asarray(mapper.pos_f[v], dtype=float) for v in mapper}


def default_layout(graph):
    """Use saved Mapper positions, computing them if any are missing."""
    mapper = graph.mapper
    if not all(v in mapper.pos_f for v in mapper):
        mapper.set_pos_from_f()
    return layout_from_mapper(mapper)


def _positions(graph, layout):
    if layout is None:
        return default_layout(graph)
    if hasattr(layout, "pos_f"):
        return layout_from_mapper(layout)
    return {v: np.asarray(layout[v], dtype=float) for v in graph.f}


def _label(vertex):
    return str(_format_node_label(vertex))


def _arcs(graph):
    """Group edges into maximal uphill paths through regular vertices."""
    mapper = graph.mapper
    regular = {v for v in mapper if mapper.in_degree(v) == mapper.out_degree(v) == 1}
    starts = sorted((edge for edge in graph.edges() if edge[0] not in regular),
                    key=lambda edge: (graph.f[edge[0]], _label(edge[0]), _label(edge[1])))
    arcs = []
    for start in starts:
        path = [start]
        while path[-1][1] in regular:
            path.append(next(iter(mapper.out_edges(path[-1][1]))))
        arcs.append(path)
    return arcs


def _styles(groups, active, prefix, color_scheme, warn_color_reuse):
    visible = [i for i, group in enumerate(groups) if active.intersection(group)]
    palette = plt.get_cmap(color_scheme)
    listed = isinstance(palette, ListedColormap)
    if listed and warn_color_reuse and len(visible) > palette.N:
        warn(f"Color scheme {palette.name!r} has {palette.N} colors for {len(visible)} source groups.",
             UserWarning, stacklevel=3)
    styles = {}
    for rank, i in enumerate(visible):
        group = groups[i]
        color = palette(rank % palette.N if listed else (rank + 0.5) / len(visible))
        label = f"{_label(group[0][0])}→{_label(group[-1][1])}"
        if prefix:
            label = f"{prefix}{i}: {label}"
        for edge in active.intersection(group):
            styles[edge] = color, label
    return styles


def _edge_position(pos, edge, parameter):
    lower, upper = edge
    return pos[lower] + float(parameter) * (pos[upper] - pos[lower])


def _point_position(graph, pos, point):
    if point[0] == "v":
        return pos[point[1]]
    _, edge, height = point
    return _edge_position(pos, edge, height - graph.f[edge[0]])


def _draw_graph(ax, graph, positions, norm):
    for u, v in graph.edges():
        xy = np.asarray([positions[u], positions[v]])
        ax.plot(*xy.T, color=INK, linewidth=1.5, zorder=1)
    for v, (x, y) in positions.items():
        color = plt.get_cmap("viridis")(norm(y)) if norm is not None else "white"
        ax.scatter(x, y, s=240, color=color, edgecolor=INK, zorder=5)
        label_color = "white" if norm is not None and norm(y) < 0.55 else INK
        ax.annotate(_label(v), (x, y), ha="center", va="center", color=label_color,
                    fontsize=8, zorder=6)
    ax.set_xticks([])
    ax.set_ylabel("Function value")
    ax.margins(x=0.12, y=0.08)


def _draw_tracks(axes, mapping, source_pos, target_pos, styles, show_breakpoints):
    source_ax, target_ax = axes
    target = mapping["target"]
    tracks = defaultdict(list)
    collapsed = defaultdict(list)
    for edge, pieces in mapping["segments"].items():
        if edge not in styles:
            continue
        color, _ = styles[edge]
        xy = np.asarray([source_pos[v] for v in edge])
        source_ax.plot(*xy.T, color=color, linewidth=3, zorder=3)
        for i, (s0, _, p0, p1) in enumerate(pieces):
            mark = show_breakpoints and i > 0
            if mark:
                source_ax.scatter(*_edge_position(source_pos, edge, s0), s=26,
                                  facecolor="white", edgecolor=color, zorder=4)
            if p0 == p1:
                collapsed[p0].append(color)
            else:
                tracks[target.common_edge(p0, p1)].append((p0, p1, color, mark))

    xs = [pos[0] for pos in target_pos.values()]
    gap = max(np.ptp(xs) * 0.006, 0.006) if xs else 0.006
    for lanes in tracks.values():
        for i, (p0, p1, color, mark) in enumerate(lanes):
            start = _point_position(target, target_pos, p0)
            end = _point_position(target, target_pos, p1)
            offset = np.array([(i - (len(lanes) - 1) / 2) * gap, 0])
            step = 0.15 * (end - start)
            xy = np.vstack((start, start + step + offset, end - step + offset, end))
            target_ax.plot(*xy.T, color=color, linewidth=2.6, zorder=3,
                           solid_capstyle="round", solid_joinstyle="round")
            if mark:
                target_ax.scatter(*start, s=26, facecolor="white", edgecolor=color, zorder=7)
    for point, colors in collapsed.items():
        position = _point_position(target, target_pos, point)
        for i, color in enumerate(colors):
            dx = 10 * (i - (len(colors) - 1) / 2)
            target_ax.annotate("●", position, xytext=(dx, -13), textcoords="offset points",
                               ha="center", va="center", color=color, zorder=8)


def _draw_witness(axes, mapping, positions, witness, direction):
    labels = ("x", "y") if direction == "phi" else ("y", "x")
    graphs = mapping["source"], mapping["target"]
    for i, (pair, marker) in enumerate(zip(witness["pair"], ("*", "D"))):
        points = pair if direction == "phi" else pair[::-1]
        xy = [_point_position(graph, pos, point)
              for graph, pos, point in zip(graphs, positions, points)]
        for ax, point, label in zip(axes, xy, labels):
            ax.scatter(*point, s=120, marker=marker, color="gold", edgecolor=INK, zorder=10)
            ax.annotate(label + ("′" if i else ""), point, xytext=(7, -12),
                        textcoords="offset points", fontsize=8, zorder=11)
        axes[1].add_artist(ConnectionPatch(xy[0], xy[1], "data", axesA=axes[0], axesB=axes[1],
                                           color=INK, linestyle=":", linewidth=1))


def draw_fd_metromaps(F, G, Phi, Psi, *, posF=None, posG=None, direction="both", color_by="arc",
                     color_scheme="tab10", warn_color_reuse=True, show_vert_maps=True,
                     level_set_colors=False, sharey=True, witness=None, only=None, only_arc=None,
                     show_breakpoints=True, figsize=None, max_legend=16, names=("F", "G"),
                     only_branch=None, label_stations=None):
    """Draw recovered maps as paired metro maps on the original graphs.

    Parameters:
        F, G (FunctionalDistortionGraph): Graphs used to build the maps.
        Phi, Psi (dict): Continuous maps from F to G and G to F.
        posF, posG (MapperGraph or dict, optional): Use a Mapper graph's
            saved positions or supply vertex-to-coordinate dictionaries.
            Defaults to the positions of the underlying Mapper graphs.
        direction (str): 'phi', 'psi', or 'both'. Defaults to 'both'.
        color_by (str): 'arc' colors maximal directed paths through regular
            vertices. 'edge' colors individual edges. 'branch' aliases 'arc'.
        color_scheme (str): Matplotlib palette. Defaults to 'tab10'.
        warn_color_reuse (bool): Warn when a listed palette repeats colors.
        show_vert_maps (bool): Label target points with their source vertices.
            The legacy keyword label_stations overrides this when supplied.
        level_set_colors (bool): Color stations by their plotted height.
        sharey (bool): Align the heights of the two panels in each row.
        witness (dict, optional): Distortion witness from functional_distortion.
        only (iterable, optional): Source edge pairs to highlight.
        only_arc (int, optional): Index of one source arc to highlight.
            The legacy keyword only_branch is also accepted.
        show_breakpoints (bool): Mark interior reparameterization breakpoints.
        figsize (tuple, optional): Figure size in inches.
        max_legend (int): Maximum legend entries per row.
        names (tuple): Display names of F and G.

    Returns:
        matplotlib.figure.Figure: Two panels per direction, source then target.

    Raises:
        ValueError: If the direction, color grouping, or arc index is invalid.
    """
    if direction not in ("both", "phi", "psi"):
        raise ValueError("direction must be 'both', 'phi', or 'psi'")
    if color_by not in ("arc", "edge", "branch"):
        raise ValueError("color_by must be 'arc' or 'edge'")
    if only_arc is not None and only_branch is not None and only_arc != only_branch:
        raise ValueError("only_arc and only_branch must select the same arc")
    arc_index = only_branch if only_arc is None else only_arc
    if label_stations is not None:
        show_vert_maps = label_stations

    positions = _positions(F, posF), _positions(G, posG)
    rows = {"phi": (Phi, positions, names), "psi": (Psi, positions[::-1], names[::-1])}
    directions = ("phi", "psi") if direction == "both" else (direction,)
    selected = set(only) if only is not None else None
    fig, axes = plt.subplots(len(directions), 2, figsize=figsize or (16, 9 * len(directions)),
                             squeeze=False, sharey="row" if sharey else False, constrained_layout=True)
    for which, row_axes in zip(directions, axes):
        mapping, (source_pos, target_pos), (source_name, target_name) = rows[which]
        source, target = mapping["source"], mapping["target"]
        arcs = _arcs(source)
        active = set(source.edges()) if selected is None else selected.intersection(source.edges())
        if arc_index is not None:
            if not 0 <= arc_index < len(arcs):
                raise ValueError(f"Source arc {arc_index} is out of range for {which}.")
            active &= set(arcs[arc_index])
        groups = [[edge] for edge in source.edges()] if color_by == "edge" else arcs
        styles = _styles(groups, active, "" if color_by == "edge" else "A", color_scheme, warn_color_reuse)
        heights = [xy[1] for pos in (source_pos, target_pos) for xy in pos.values()]
        norm = Normalize(min(heights), max(heights)) if heights and level_set_colors else None
        _draw_graph(row_axes[0], source, source_pos, norm)
        _draw_graph(row_axes[1], target, target_pos, norm)
        _draw_tracks(row_axes, mapping, source_pos, target_pos, styles, show_breakpoints)
        if show_vert_maps:
            images = defaultdict(list)
            for vertex, point in mapping["vertex_image"].items():
                images[point].append(_label(vertex))
            for point, labels in images.items():
                row_axes[1].annotate("← " + ", ".join(labels), _point_position(target, target_pos, point),
                                     xytext=(9, 8), textcoords="offset points", fontsize=7, zorder=8)
        symbol = "Φ" if which == "phi" else "Ψ"
        row_axes[0].set_title(f"{symbol}: {source_name} → {target_name}")
        row_axes[1].set_title(target_name)
        legend = dict.fromkeys(styles.values())
        handles = [Line2D([], [], color=color, linewidth=3, label=label)
                   for color, label in list(legend)[:max_legend]]
        if handles:
            title = "Source edges" if color_by == "edge" else "Source arcs"
            row_axes[1].legend(handles=handles, title=title, loc="best", fontsize=7)
        if witness is not None:
            _draw_witness(row_axes, mapping, (source_pos, target_pos), witness, which)
    if witness is not None:
        df, dg = witness["d_f"], witness["d_g"]
        fig.suptitle(f"Distortion witness  |  d_f = {df}, d_g = {dg}, D = {abs(df - dg) / 2} (grid units)")
    return fig
