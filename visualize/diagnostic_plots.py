"""Standalone figures for edge reparameterizations and distortion squares."""

from fractions import Fraction

import matplotlib.pyplot as plt

from core.distortion import _lines, _square_max, candidate_points, graph_distance, side_point


def plot_edge_reparameterization(mapping, source_edge, *, target_edge=None, ax=None):
    """Plot how an existing map traverses its target route along one source edge.

    Parameters:
        mapping (dict): Map returned by ``map_to`` or the FD map builder.
        source_edge (tuple): Uphill endpoint pair of the source edge.
        target_edge (tuple, optional): Uphill target edge to highlight on
            the route. All pieces of the route remain visible.
        ax (matplotlib.axes.Axes, optional): Axes to draw on.

    Returns:
        matplotlib.axes.Axes: Source parameter on the horizontal axis and
            height in grid units on the vertical axis.
    """
    source, target = mapping["source"], mapping["target"]
    pieces = mapping["segments"][source_edge]
    if target_edge is not None and not target.has_edge(*target_edge):
        raise ValueError(f"Unknown uphill target edge {target_edge!r}")
    if ax is None:
        _, ax = plt.subplots(layout="constrained")

    times = [pieces[0][0], *(piece[1] for piece in pieces)]
    heights = [target.height(pieces[0][2]), *(target.height(piece[3]) for piece in pieces)]
    lower = source.f[source_edge[0]]
    ax.plot([0, 1], [lower, lower + 1], "--", color="#555555", label="Source height")
    ax.plot(times, heights, "o-", color="#2166ac", label="Mapped height")
    if target_edge is not None:
        label = f"Target edge {target_edge[0]} → {target_edge[1]}"
        for start, end, p0, p1 in pieces:
            if target_edge in target.edges_of(p0) and target_edge in target.edges_of(p1):
                ax.plot([start, end], [target.height(p0), target.height(p1)],
                        color="#d55e00", linewidth=2.5, label=label)
                label = None
    ax.set(xlabel="Source edge parameter", ylabel="Height (grid units)",
           title=f"Reparameterization of {source_edge[0]} → {source_edge[1]}")
    ax.legend()
    return ax


def plot_distortion_square(F, G, A, B, *, samples=41, ax=None):
    """Draw the distortion square for two correspondence pieces.

    Parameters:
        F, G (FunctionalDistortionGraph): Graphs used to build the maps.
        A, B (CSegment): Pieces from ``c_segments(Phi, Psi)`` or the
            ``segA`` and ``segB`` entries of a distortion witness.
        samples (int, optional): Grid points per axis for the background.
        ax (matplotlib.axes.Axes, optional): Axes to draw on.

    Returns:
        matplotlib.axes.Axes: The sampled discrepancy field with exact
            partition lines, candidate points, and a maximizing candidate.
    """
    if samples < 2:
        raise ValueError("samples must be at least 2")
    if ax is None:
        _, ax = plt.subplots(layout="constrained")

    grid = [Fraction(i, samples - 1) for i in range(samples)]
    points_a = [(side_point(F, A.F, s), side_point(G, A.G, s)) for s in grid]
    points_b = [(side_point(F, B.F, t), side_point(G, B.G, t)) for t in grid]
    field = [[float(abs(graph_distance(F, x, xx) - graph_distance(G, y, yy)) / 2)
              for x, y in points_a] for xx, yy in points_b]
    coordinates = list(map(float, grid))
    mesh = ax.contourf(coordinates, coordinates, field, levels=16, cmap="viridis")
    ax.figure.colorbar(mesh, ax=ax, label=r"$|d_f-d_g|/2$ (grid units)")

    candidates = candidate_points(A, B)
    for a, b, c in _lines(A, B):
        on_line = [(s, t) for s, t in candidates if a * s + b * t == c]
        start, end = min(on_line), max(on_line)
        ax.plot(*zip(start, end), color="white", linewidth=0.9, alpha=0.8)
    ax.scatter(*zip(*candidates), s=22, facecolor="white", edgecolor="#222222",
               zorder=3, label=f"{len(candidates)} candidates")
    gap, witness = _square_max(F, G, A, B, candidates)
    ax.scatter(witness["sigma"], witness["tau"], marker="*", s=125,
               color="#f0b429", edgecolor="#222222", zorder=4, label="Exact maximum")
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xlabel=r"Piece A parameter $\sigma$",
           ylabel=r"Piece B parameter $\tau$", title=f"Distortion square · maximum {gap / 2}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    return ax
