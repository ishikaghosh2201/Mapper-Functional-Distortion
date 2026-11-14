"""Recover continuous maps between the original graphs of a ceREEBerus interleaving.

Start with ``maps_from_interleave`` after fitting a ceREEBerus ``Interleave``.
The interleaving maps into smoothed graphs. This module decodes their
components into sets of vertices in the original graphs, then uses those
sets to choose vertex images and route each edge in both directions.

For a source edge ``(u, v)``, the route runs from the image of ``u`` to a
handoff vertex inside the lower endpoint's component, then to the image
of ``v`` inside the upper endpoint's component. A Fréchet reparameterization
sets where along the source edge each target vertex is reached. Explicit assignments are checked before map recovery. At ``n = 0``, recovery checks and uses the height-preserving isomorphisms directly.
"""
import random
from copy import deepcopy
from bisect import bisect_left, bisect_right
from fractions import Fraction
from numbers import Integral

import networkx as nx

from core.functionaldistortiongraph import FunctionalDistortionGraph


# ---------------------------------------------------------------------------
# Read and validate ceREEBerus interleavings
# ---------------------------------------------------------------------------
def _edge_id(G, e):
    """ceREEBerus edge (u, v, key) -> our edge id (lower, upper)."""
    u, v = e[0], e[1]
    return (u, v) if G.f[v] - G.f[u] == 1 else (v, u)


def _block_images(block, level, name, *, strict=True):
    """Read the image of each column from a ceREEBerus matrix block."""
    for index, obj in enumerate(block.cols):
        entries = block.array[:, index]
        selected = entries == 1 if strict else entries > 0.5
        if selected.sum() != 1 or (strict and not ((entries == 0) | selected).all()):
            raise ValueError(f"{name}: {obj!r} must have exactly one image at level {level}")
        yield obj, block.rows[selected.argmax()]


def _decode(matrix, components, name):
    return {
        obj: components[image]
        for level in matrix.get_all_block_indices()
        for obj, image in _block_images(matrix[level], level, name, strict=False)
    }


def _assignment_labels(graph, kind):
    return graph.func_to_vertex_dict() if kind == "V" else graph.func_to_edge_dict()


def _read_set_map(matrix, source, target, kind, name):
    """Read a total, height-preserving set map from its labeled blocks."""
    source_levels = _assignment_labels(source, kind)
    target_levels = _assignment_labels(target, kind)
    images = {}
    levels = set(matrix.get_all_block_indices())
    for level in sorted(levels | source_levels.keys()):
        if level not in levels:
            raise ValueError(f"{name}: missing block at level {level}")
        block = matrix[level]
        columns = source_levels.get(level, ())
        rows = target_levels.get(level, ())
        if (len(block.cols) != len(columns) or set(block.cols) != set(columns)
                or len(block.rows) != len(rows) or set(block.rows) != set(rows)):
            raise ValueError(f"{name}: incorrect source or target labels at level {level}")
        images.update(_block_images(block, level, name))
    return images


def _check_interleaving_graphs(F, G, n):
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    if F.delta != G.delta:
        raise ValueError("F and G must have the same delta")


def validate_interleaving(assignment):
    """Raise ValueError if an Assignment does not define an exact interleaving.

    Checks all eight maps for one image per object at the correct grid
    level, then checks edge incidence, compatibility with the induced
    smoothing maps, and both interleaving triangles on vertices and edges.
    """
    _check_interleaving_graphs(assignment.F("0"), assignment.G("0"), assignment.n)
    directions = (("phi", assignment.F, assignment.G), ("psi", assignment.G, assignment.F))
    images = {}
    for direction, source, target in directions:
        for level, target_level in (("0", "n"), ("n", "2n")):
            for kind in ("V", "E"):
                name = f"{direction}[{level}][{kind}]"
                images[direction, level, kind] = _read_set_map(
                    assignment.get_interleaving_map(direction, level, kind),
                    source(level), target(target_level), kind, name,
                )
            vertices = images[direction, level, "V"]
            for edge, image in images[direction, level, "E"].items():
                if (vertices[edge[0]], vertices[edge[1]]) != image[:2]:
                    raise ValueError(f"{direction}[{level}]: edge incidence fails at {edge!r}")

    induced = {}
    for graph_name, graph in (("F", assignment.F), ("G", assignment.G)):
        for level, target_level in (("0", "n"), ("n", "2n")):
            for kind in ("V", "E"):
                induced[graph_name, level, kind] = _read_set_map(
                    assignment.I(graph_name, level, kind),
                    graph(level), graph(target_level), kind, f"I_{graph_name}[{level}][{kind}]",
                )

    for direction, reverse, source, target in (("phi", "psi", "F", "G"), ("psi", "phi", "G", "F")):
        for kind in ("V", "E"):
            for obj, image in images[direction, "0", kind].items():
                smoothed = induced[source, "0", kind][obj]
                if images[direction, "n", kind][smoothed] != induced[target, "n", kind][image]:
                    raise ValueError(f"{direction}: smoothing compatibility fails for {kind} {obj!r}")
                if images[reverse, "n", kind][image] != induced[source, "n", kind][smoothed]:
                    raise ValueError(f"{direction}: interleaving triangle fails for {kind} {obj!r}")


def _smoothing_components(assignment, graph):
    """Return the original vertex sets for the vertices and edges of a smoothing.

    A smoothed vertex at height ``t`` comes from a component of the closed
    slice [t - n, t + n]. An edge from ``t`` to ``t + 1`` comes from a slice
    centered at t + 1/2. Match components in ceREEBerus's construction
    order and check the matches against the induced smoothing maps.
    """
    graph_at = getattr(assignment, graph)
    source, smooth = graph_at("0"), graph_at("n")
    n = assignment.n
    if n < 1:
        raise ValueError("Map construction requires a positive interleaving parameter.")
    original = set(source.nodes)

    def slice_components(center):
        sliced = source.slice(center - n, center + n, type="closed")
        return [frozenset(component & original) for component in sliced.connected_components()]

    vertices = {}
    edges = {}
    labels_by_level = smooth.func_to_vertex_dict()
    edges_by_level = smooth.func_to_edge_dict()
    for height, labels in labels_by_level.items():
        components = slice_components(height)
        if len(labels) != len(components) or any(not component for component in components):
            raise ValueError(f"Cannot match smoothing vertices at height {height}.")
        vertices.update(zip(labels, components))

    for height, labels in edges_by_level.items():
        # Group keyed edges by their endpoints, preserving the insertion order
        # within each pair so parallel edges retain their component labels.
        edges_by_endpoints = {}
        for edge in labels:
            edges_by_endpoints.setdefault(edge[:2], []).append(edge)
        components_by_endpoints = {}
        for component in slice_components(height + 0.5):
            lower = [v for v in labels_by_level[height] if component <= vertices[v]]
            upper = [v for v in labels_by_level[height + 1] if component <= vertices[v]]
            if not component or len(lower) != 1 or len(upper) != 1:
                raise ValueError(f"Cannot match smoothing edge endpoints at height {height}.")
            pair = (lower[0], upper[0])
            components_by_endpoints.setdefault(pair, []).append(component)
        if edges_by_endpoints.keys() != components_by_endpoints.keys():
            raise ValueError(f"Recovered components do not cover smoothing edges at height {height}.")
        for pair, keyed_edges in edges_by_endpoints.items():
            components = components_by_endpoints[pair]
            if len(keyed_edges) != len(components):
                raise ValueError(f"Cannot match parallel smoothing edges between {pair!r}.")
            edges.update(zip(keyed_edges, components))

    for kind, components in (("V", vertices), ("E", edges)):
        induced = _read_set_map(assignment.I(graph, "0", kind), source, smooth, kind,
                                f"I_{graph}[0][{kind}]")
        for obj, image in induced.items():
            members = (obj,) if kind == "V" else obj[:2]
            if not components[image].issuperset(members):
                raise ValueError(f"Recovered smoothing labels disagree with the induced {kind} map at {obj!r}.")
    return {"V": vertices, "E": edges}


def decode_interleaving(assignment):
    """Convert interleaving images to candidate vertex sets on the original graphs.

    Parameters:
        assignment (Assignment):
            A ceREEBerus assignment with vertex and edge maps in both directions.

    Returns:
        tuple:
            ``(F, G, phi_v, phi_e, psi_v, psi_e)``. The first two entries are
            continuous graph wrappers. Each remaining entry maps a source
            vertex or uphill edge pair to a frozenset of target vertices.
            For zero smoothing, a vertex gives a singleton and an edge
            gives its two endpoints.
    """
    F = FunctionalDistortionGraph(assignment.F("0"))
    G = FunctionalDistortionGraph(assignment.G("0"))

    def components_for(graph):
        if assignment.n != 0:
            return _smoothing_components(assignment, graph)
        # Zero smoothing leaves every vertex and edge in place.
        original = getattr(assignment, graph)("0")
        return {"V": {v: frozenset((v,)) for v in original.nodes},
                "E": {edge: frozenset(edge[:2]) for edge in original.edges(keys=True)}}

    components_F = components_for("F")
    components_G = components_for("G")

    def decode_direction(name, mapping, source, components):
        vertices = _decode(mapping("0", "V"), components["V"], f"{name}[0][V]")
        edges = _decode(mapping("0", "E"), components["E"], f"{name}[0][E]")
        return vertices, {_edge_id(source, edge): members for edge, members in edges.items()}

    phi_v, phi_e = decode_direction("phi", assignment.phi, F, components_G)
    psi_v, psi_e = decode_direction("psi", assignment.psi, G, components_F)
    return F, G, phi_v, phi_e, psi_v, psi_e


# ---------------------------------------------------------------------------
# the construction
# ---------------------------------------------------------------------------
def _check_construction_options(points, paths):
    if points not in ("random", "close", "joint"):
        raise ValueError("points must be 'random', 'close', or 'joint'")
    if paths not in ("fewest_edges", "min_span"):
        raise ValueError("paths must be 'fewest_edges' or 'min_span'")


def _ordered(vertices):
    return sorted(vertices, key=repr)


def _near(candidates, T, target_height):
    """Return the closest-height candidates in a fixed order."""
    cands = _ordered(candidates)
    if not cands:
        raise ValueError(f"No candidate vertices for height {target_height}")
    best = min(abs(T.f[c] - target_height) for c in cands)
    # With slack enabled, the cutoff would be best + slack.
    return [c for c in cands if abs(T.f[c] - target_height) == best]


def _pick_nearest(rng, candidates, T, target_height):
    """Uniformly among the candidates whose height is closest to target_height."""
    return rng.choice(_near(candidates, T, target_height))


def _random_shortest_path(rng, T, allowed, a, b):
    """A random fewest-edges path a -> b using only vertices in `allowed` (a, b included).
    With rng=None the first neighbour (in a fixed order) is taken at each step: deterministic."""
    if a == b:
        return [a]
    subgraph = T.mapper.to_undirected(as_view=True).subgraph(allowed)
    dist = nx.single_source_shortest_path_length(subgraph, b) if b in subgraph else {}
    if a not in dist:
        raise ValueError(f"no path from {a!r} to {b!r} inside the given set; is the set connected?")
    path = [a]
    while path[-1] != b:
        x = path[-1]
        nxt = _ordered(y for y in subgraph[x] if dist[y] == dist[x] - 1)
        path.append(rng.choice(nxt) if rng is not None else nxt[0])
    return path


def _min_span_route(target, lower_set, upper_set, start, handoff, end, cache):
    """Find both paths in a minimum-width shared height window.

    At that width, choose the route with the fewest total edges. Equal
    lengths are resolved by window order and deterministic BFS tie-breaking.
    """
    key = (frozenset(lower_set), frozenset(upper_set), start, handoff, end)
    if key in cache:
        return list(cache[key])

    heights = target.f
    anchors = [heights[v] for v in (start, handoff, end)]
    low, high = min(anchors), max(anchors)
    levels = sorted({heights[v] for v in lower_set | upper_set})
    windows = sorted((hi - lo, lo, hi) for lo in levels if lo <= low
                     for hi in levels if hi >= high)
    best_route = None
    for width, lo, hi in windows:
        if best_route is not None and width > best_width:
            break
        lower_band = {v for v in lower_set if lo <= heights[v] <= hi}
        upper_band = {v for v in upper_set if lo <= heights[v] <= hi}
        try:
            lower = _random_shortest_path(None, target, lower_band, start, handoff)
            upper = _random_shortest_path(None, target, upper_band, handoff, end)
        except ValueError:
            continue
        route = lower + upper[1:]
        if best_route is None or len(route) < len(best_route):
            best_route, best_width = route, width

    if best_route is None:
        raise ValueError(f"no route from {start!r} through {handoff!r} to {end!r} inside the given sets")
    cache[key] = tuple(best_route)
    return best_route


def _edge_route(target, endpoint_sets, edge, start, handoff, end, rng, paths, cache):
    """Join two paths at a vertex shared by the endpoint components."""
    u, v = edge
    if handoff not in endpoint_sets[u] or handoff not in endpoint_sets[v]:
        raise ValueError(f"edge {edge!r}: handoff vertex {handoff!r} must belong to both endpoint sets")
    lower_set, upper_set = endpoint_sets[u], endpoint_sets[v]
    if paths == "min_span":
        return _min_span_route(target, lower_set, upper_set, start, handoff, end, cache)
    lower = _random_shortest_path(rng, target, lower_set, start, handoff)
    upper = _random_shortest_path(rng, target, upper_set, handoff, end)
    return lower + upper[1:]


def _route_cost(target, route):
    heights = [target.f[v] for v in route]
    span = max(heights) - min(heights)
    return abs(span - 1), span, len(route) - 1


ETA = Fraction(1, 1000)   # Perturbation used to separate route vertices that get the same time
GAP = ETA                 # compatibility alias


def frechet_times(t, heights):
    """Compute a Fréchet reparameterization for one source edge.

    Parameters:
        t (int):
            Height of the lower endpoint. The source edge has height ``t + s``
            at parameter ``s`` in [0, 1].
        heights (sequence):
            Exact heights of the target route vertices in traversal order.
            Supply one height for a constant route.

    Returns:
        tuple:
            ``(times, d_e, spread)``. The times are Fractions at which the
            source edge reaches each route vertex. They start at 0 and end
            at 1, with strictly increasing interior values. A constant
            route returns just [0]. The ideal height error is ``d_e``.
            ``spread`` is true if equal times had to be separated.

    The ideal error is the maximum of the endpoint mismatches, half the
    largest drop along the route, and the largest excursion outside
    [t, t + 1]. At that tolerance, each route vertex has an interval of
    feasible parameters. A backward pass computes upper bounds and a
    forward pass chooses nondecreasing parameters near matching heights. If times
    coincide, blending with equally spaced parameters by ``ETA`` makes
    them strictly increasing and increases the error by at most ``ETA``.

    Examples:
        >>> frechet_times(0, [0, 1])
        ([Fraction(0, 1), Fraction(1, 1)], Fraction(0, 1), False)
    """
    t = Fraction(t)
    h = [Fraction(x) for x in heights]
    k = len(h) - 1
    if not h:
        raise ValueError("A route must have at least one vertex")
    drop, top = Fraction(0), h[0]
    for x in h[1:]:                  # largest h_i - h_j over i < j, in one pass
        drop = max(drop, top - x)
        top = max(top, x)
    outside = max(max(t - x, x - (t + 1)) for x in h)          # how far a vertex lies outside the band
    d = max(abs(h[0] - t), abs(h[k] - (t + 1)), drop / 2, outside, Fraction(0))
    if k <= 1:
        return [Fraction(0)] + ([Fraction(1)] if k == 1 else []), d, False

    upper = [min(Fraction(1), x - t + d) for x in h]
    for i in range(k - 1, -1, -1):
        upper[i] = min(upper[i], upper[i + 1])

    times = [Fraction(0)]
    for i in range(1, k):
        # The previous feasible time already bounds all earlier lower limits.
        low = max(times[-1], h[i] - t - d)
        if low > upper[i]:
            raise ValueError(f"no timing within {d} for route heights {h} on edge [{t}, {t + 1}]")
        times.append(min(max(h[i] - t, low), upper[i]))
    times.append(Fraction(1))
    spread = any(right <= left for left, right in zip(times, times[1:]))
    if spread:
        times = [(1 - ETA) * time + ETA * Fraction(i, k) for i, time in enumerate(times)]
    return times, d, spread


def _handoff(route, times, set_u, set_v):
    """
    (lower_ok, upper_ok): does the image of the lower half s in [0,1/2] of the edge stay inside
    phi_u, and of the upper half s in [1/2,1] inside phi_v?  The image of a half is the route
    from its first vertex up to the point reached at s = 1/2; it lies in the (induced) set iff
    all route vertices reached in that half are in it and, if s = 1/2 falls strictly inside a
    step R_j -> R_{j+1}, also the vertex at the other end of that step.
    """
    half = Fraction(1, 2)
    if len(route) == 1:
        return route[0] in set_u, route[0] in set_v
    # A half that ends in an edge interior touches that entire target edge.
    lower_end = bisect_left(times, half)
    upper_start = bisect_right(times, half) - 1
    return (all(v in set_u for v in route[:lower_end + 1]),
            all(v in set_v for v in route[upper_start:]))


def handoff_report(mapping):
    """Return the lower and upper half checks for each edge of either map."""
    return {edge: (info["lower_in_set"], info["upper_in_set"])
            for edge, info in mapping["route_info"].items()}


def failing_handoffs(mapping):
    """Return edges whose image leaves an endpoint's allowed set on its half."""
    return {edge: checks for edge, checks in handoff_report(mapping).items() if not all(checks)}


class _MapChooser:
    """
    Choices for one map  source -> target  (the vertex images and the hand-off vertices w_e) made so
    that every edge is sent to a route that looks like one edge (small height span).  Candidates are
    always vertices of the right set at the nearest height, so the height errors at
    vertices are no larger than with the random rule.
    """

    def __init__(self, source, target, set_v, set_e, rng, paths, cache):
        self.source, self.target, self.set_v, self.set_e = source, target, set_v, set_e
        self.rng, self.paths, self.cache = rng, paths, cache
        self.near = {v: _near(set_v[v], target, source.f[v]) for v in source.f}
        self.img = {v: rng.choice(self.near[v]) for v in source.f}
        self.wcand = {e: _near(set_e[e], target, source.f[e[0]] + Fraction(1, 2))
                      for e in source.edges()}

    def best_w(self, e, a, b):
        scored = []
        for w in self.wcand[e]:
            route = _edge_route(self.target, self.set_v, e, a, w, b, None, self.paths, self.cache)
            scored.append((_route_cost(self.target, route), repr(w), w))
        cost, _, vertex = min(scored, key=lambda item: item[:2])
        return cost, vertex

    def vertex_key(self, v, c, extra=None):
        costs = []
        for e in self.source.incident_edges(v):
            a, b = (c, self.img[e[1]]) if e[0] == v else (self.img[e[0]], c)
            costs.append(self.best_w(e, a, b)[0])
        r = max(costs) if costs else (0, 0, 0)
        x = extra(v, c) if extra else 0
        return (max(r[0], x), x, r[1], r[2], sum(k[0] for k in costs))

    def descend(self, extra=None, sweeps=8):
        """Re-choose each image to minimize its key (ties: random) until nothing changes."""
        any_change = False
        for _ in range(sweeps):
            changed = False
            for v in self.source.f:
                scored = [(self.vertex_key(v, c, extra), c) for c in self.near[v]]
                best = min(k for k, _ in scored)
                ties = [c for k, c in scored if k == best]
                if self.img[v] not in ties:
                    self.img[v] = self.rng.choice(ties)
                    changed = any_change = True
            if not changed:
                break
        return any_change

    def w_choice(self):
        return {e: self.best_w(e, self.img[e[0]], self.img[e[1]])[1] for e in self.source.edges()}


def _joint_choices(F, G, phi_v, phi_e, psi_v, psi_e, rng, paths):
    """
    points='joint': as 'close' for both maps, and in addition Phi and Psi are chosen to be nearly
    inverse at the vertices: d_f(v, Psi(Phi(v))) and d_g(y, Phi(Psi(y))) are kept small, because
    D compares the pairs (v, Phi(v)) and (Psi(y), y) of C.
    """
    from core.distortion import graph_distance as dist
    cF = _MapChooser(F, G, phi_v, phi_e, rng, paths, {})
    cG = _MapChooser(G, F, psi_v, psi_e, rng, paths, {})
    cF.descend()
    cG.descend()

    def roundtrip_cost(forward, reverse):
        def cost(vertex, image):
            errors = [dist(forward.source, ("v", vertex), ("v", reverse.img[image]))]
            errors.extend(dist(forward.target, ("v", other), ("v", image))
                          for other, preimage in reverse.img.items() if preimage == vertex)
            return max(errors)

        return cost

    cost_phi = roundtrip_cost(cF, cG)
    cost_psi = roundtrip_cost(cG, cF)
    for _ in range(6):
        ch1 = cF.descend(extra=cost_phi, sweeps=1)
        ch2 = cG.descend(extra=cost_psi, sweeps=1)
        if not (ch1 or ch2):
            break
    return cF, cG


def build_map(source, target, set_v, set_e, rng, points="random", paths="min_span",
              chosen=None):
    """Build one continuous map from decoded interleaving components.

    Parameters:
        source, target (FunctionalDistortionGraph):
            The original graphs in the direction being built.
        set_v, set_e (dict):
            Candidate target vertices for each source vertex and each
            uphill source edge. An edge's handoff must belong to both
            endpoint sets.
        rng (random.Random):
            Generator for choosing among tied candidates and shortest paths.
        points, paths:
            Selection options documented in ``maps_from_assignment``.
        chosen (tuple, optional):
            Preselected vertex images and edge handoffs, used by the
            joint strategy.

    Returns:
        dict:
            The map from ``source.map_to``, with ``route_info`` keyed by source
            edge. Each entry records the target route, its source parameters,
            handoff vertex, ideal and realized height errors, whether tied
            parameters were separated, and the two handoff checks.
    """
    _check_construction_options(points, paths)
    cache = {}
    half = Fraction(1, 2)   # only used to pick w_e: nearest to the middle height of the edge
    if chosen is not None:
        img, w_of = chosen
    elif points == "close":
        ch = _MapChooser(source, target, set_v, set_e, rng, paths, cache)
        ch.descend()
        img, w_of = ch.img, ch.w_choice()
    else:
        img = {v: _pick_nearest(rng, set_v[v], target, source.f[v]) for v in source.f}
        w_of = None

    routes, reparameterizations, info = {}, {}, {}
    for e in source.edges():
        u, v = e
        allowed = set_e[e]
        if not allowed:
            raise ValueError(f"edge {e!r}: empty set")
        if w_of is not None:
            w = w_of[e]
        else:
            w = _pick_nearest(rng, allowed, target, source.f[u] + half)
        route = _edge_route(target, set_v, e, img[u], w, img[v], rng, paths, cache)
        t = source.f[u]
        times, d_e, spread = frechet_times(t, [target.f[x] for x in route])
        routes[e] = [("v", x) for x in route[1:-1]]
        reparameterizations[e] = times[1:-1]
        lower_ok, upper_ok = _handoff(route, times, set_v[u], set_v[v])
        at = list(zip(times, route)) if len(route) > 1 else [(Fraction(0), route[0]), (Fraction(1), route[0])]
        err = max(abs(t + s - target.f[x]) for s, x in at)   # height error at the route vertices
        info[e] = {"route": route, "times": times, "w": w, "d_e": d_e, "spread": spread,
                   "realized_error": err, "lower_in_set": lower_ok, "upper_in_set": upper_ok}
    mapping = source.map_to(target, {x: ("v", y) for x, y in img.items()}, routes, reparameterizations)
    mapping["route_info"] = info
    return mapping


def _isomorphism_map(source, target, set_v, set_e):
    """
    The map for n = 0:  set_v[v] = {image vertex},  set_e[e] = the two ends of the image edge.
    Checks that this is a height-preserving graph isomorphism (one-to-one on vertices and on
    edges, edges onto edges) and raises ValueError otherwise; nothing is guessed.
    Every edge maps directly to its image edge. The map stores the same
    route diagnostics as ``build_map``, with ``w = None``.
    """
    img = {}
    for v in source.f:
        if len(set_v[v]) != 1:
            raise ValueError(f"n = 0: vertex {v!r} is sent to {len(set_v[v])} vertices, expected 1")
        (y,) = set_v[v]
        if target.f[y] != source.f[v]:
            raise ValueError(f"n = 0: vertex {v!r} (height {source.f[v]}) is sent to "
                             f"{y!r} (height {target.f[y]})")
        img[v] = y
    if len(set(img.values())) != len(img) or len(img) != len(target.f):
        raise ValueError("n = 0: the vertex map is not one-to-one and onto")
    if len(source.edges()) != len(target.edges()):
        raise ValueError("n = 0: the graphs have different numbers of edges")
    info = {}
    for e in source.edges():
        u, v = e
        if not target.has_edge(img[u], img[v]) or set_e[e] != frozenset((img[u], img[v])):
            raise ValueError(f"n = 0: edge {e!r} is not sent to the edge between the images of its ends")
        info[e] = {"route": [img[u], img[v]], "times": [Fraction(0), Fraction(1)], "w": None,
                   "d_e": Fraction(0), "spread": False, "realized_error": Fraction(0),
                   "lower_in_set": True, "upper_in_set": True}
    mapping = source.map_to(target, {x: ("v", y) for x, y in img.items()})
    mapping["route_info"] = info
    return mapping


def maps_from_explicit_interleaving(assignment, phi, psi, *, seed=0, points="random",
                                    paths="min_span"):
    """Build continuous maps from user-supplied interleaving dictionaries.

    Parameters:
        assignment (Assignment):
            A ceREEBerus Assignment constructed with the original graphs
            and the desired n. Use its F(key) and G(key) methods to look
            up labels in the original and smoothed graphs. All eight map
            matrices are filled on a copy of this assignment.
        phi, psi (dict):
            Each dictionary has keys "0" and "n", with a "V" vertex map
            and an "E" edge map under each key. Supply source-to-target
            dictionaries with vertex labels and uphill edge triples
            (lower_vertex, upper_vertex, key), as returned by edges(keys=True).
        seed, points, paths:
            Construction options described in maps_from_assignment.

    Returns:
        dict:
            Both original continuous graphs and the recovered maps.
    """
    candidate = deepcopy(assignment)
    for direction, supplied in (("phi", phi), ("psi", psi)):
        if set(supplied) != {"0", "n"}:
            raise ValueError(f"{direction}: expected keys '0' and 'n'")
        for level, maps in supplied.items():
            if set(maps) != {"V", "E"}:
                raise ValueError(f"{direction}[{level}]: expected keys 'V' and 'E'")
            for kind, mapping in maps.items():
                name = f"{direction}[{level}][{kind}]"
                matrix = candidate.get_interleaving_map(direction, level, kind)
                columns = set()
                for height in matrix.get_all_block_indices():
                    block = matrix[height]
                    block.array.fill(0)
                    rows = {obj: index for index, obj in enumerate(block.rows)}
                    columns.update(block.cols)
                    for index, obj in enumerate(block.cols):
                        if obj not in mapping or mapping[obj] not in rows:
                            raise ValueError(f"{name}: {obj!r} needs an image at level {height}")
                        block.array[rows[mapping[obj]], index] = 1
                if extra := mapping.keys() - columns:
                    raise ValueError(f"{name}: unknown source objects {extra!r}")
    return maps_from_assignment(candidate, seed=seed, points=points, paths=paths)


def maps_from_assignment(assignment, seed=0, points="random", paths="min_span", *, validate=True):
    """Recover both continuous maps from a ceREEBerus Assignment.

    Parameters:
        assignment (Assignment):
            A valid assignment with interleaving parameter ``n``.
        seed (int, optional):
            Seed for reproducible choices. Defaults to 0.
        points (str, optional):
            ``"random"`` chooses among vertices closest in height.
            ``"close"`` also scores the height span of incident routes.
            ``"joint"`` also scores vertex round-trip errors across both
            maps. Defaults to ``"random"``.
        paths (str, optional):
            ``"fewest_edges"`` minimizes the number of edges in each path.
            ``"min_span"`` minimizes the height span of the entire route
            through the handoff, then its total edge count. Defaults to
            ``"min_span"``.
        validate (bool, optional):
            Check the full interleaving diagrams before recovery. Defaults to True.

    Returns:
        dict:
            ``F`` and ``G`` are the continuous graphs. ``Phi`` and ``Psi``
            are the recovered maps in the two directions. The dictionary
            also contains ``n``, the construction options, and the decoded
            vertex and edge candidate sets for both maps.

    At ``n = 0``, the assignment determines both maps directly, so the
    construction options have no effect.
    """
    _check_construction_options(points, paths)
    if validate:
        validate_interleaving(assignment)
    F, G, phi_v, phi_e, psi_v, psi_e = decode_interleaving(assignment)
    if assignment.n == 0:
        Phi = _isomorphism_map(F, G, phi_v, phi_e)
        Psi = _isomorphism_map(G, F, psi_v, psi_e)
        if any(Psi["vertex_image"][Phi["vertex_image"][v][1]] != ("v", v) for v in F.f):
            raise ValueError("n = 0: psi is not the inverse of phi")
    else:
        rng = random.Random(seed)
        choices = (None, None)
        if points == "joint":
            choosers = _joint_choices(F, G, phi_v, phi_e, psi_v, psi_e, rng, paths)
            choices = [(chooser.img, chooser.w_choice()) for chooser in choosers]
        Phi = build_map(F, G, phi_v, phi_e, rng, points, paths, chosen=choices[0])
        Psi = build_map(G, F, psi_v, psi_e, rng, points, paths, chosen=choices[1])
    return {"F": F, "G": G, "Phi": Phi, "Psi": Psi, "n": assignment.n, "seed": seed,
            "points": points, "paths": paths,
            "phi_v": phi_v, "phi_e": phi_e, "psi_v": psi_v, "psi_e": psi_e}


def maps_from_interleave(interleave, seed=0, *, validate=False, **options):
    """Recover continuous maps from a fitted ceREEBerus Interleave.

    Parameters:
        interleave (Interleave):
            A ceREEBerus object with an assignment from ``fit()``.
        seed (int, optional):
            Seed for reproducible choices. Defaults to 0.
        validate (bool, optional):
            Require a full exact interleaving. Defaults to False for
            compatibility with fitted assignments from ceREEBerus versions
            that leave some smoothed-map columns empty.
        **options:
            Forwarded to ``maps_from_assignment``.

    Returns:
        dict:
            The result from ``maps_from_assignment``. Use ``F``, ``G``,
            ``Phi``, and ``Psi`` from this result together when computing
            functional distortion.

    Raises:
        ValueError:
            If the Interleave object has no fitted assignment.
    """
    if interleave.assignment is None:
        raise ValueError("call interleave.fit() first")
    return maps_from_assignment(interleave.assignment, seed=seed, validate=validate, **options)
