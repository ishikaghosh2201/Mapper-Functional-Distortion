"""Measure the functional distortion of two continuous maps between Mapper graphs.

Use ``functional_distortion`` to compare a pair of maps in opposite directions.
The objective is the maximum of their two height errors and half the largest
height-span distance discrepancy over the correspondence induced by the maps.
Calculations use exact fractions in grid units. The public result also gives
values scaled by the graphs' shared ``delta``.
"""
import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from itertools import chain
from fractions import Fraction
from typing import Any, Optional

from core.functionaldistortiongraph import evaluate_map


# ---------------------------------------------------------------------------
# d_f between two points of one graph
# ---------------------------------------------------------------------------
def graph_distance_simple(G, p, q):
    """
    d_f(p, q): the smallest length b - a of an interval [a, b] such that p and q are
    connected inside f^{-1}[a, b].  Simple version (union-find per candidate interval);
    kept as a cross-check of the fast version graph_distance().
    """
    p, q = G.normalize(p), G.normalize(q)
    if p == q:
        return Fraction(0)
    hp, hq = G.height(p), G.height(q)
    lo, hi = min(hp, hq), max(hp, hq)
    if G.common_edge(p, q) is not None:
        return hi - lo
    levels = sorted(set(G.f.values()))
    lows = [lo] + [t for t in levels if t < lo]
    highs = [hi] + [t for t in levels if t > hi]  # ascending
    best = None
    for a in lows:
        for b in highs:
            if best is not None and b - a >= best:
                break  # b only grows from here
            if _connected(G, p, q, a, b):
                best = b - a
                break
    if best is None:
        raise ValueError(f"{p} and {q} are in different components of the graph")
    return best


def _connected(G, p, q, a, b):
    """Are p and q connected inside f^{-1}[a, b]?"""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    inside = lambda v: a <= G.f[v] <= b
    for lo_v, hi_v in G.edges():
        if inside(lo_v) and inside(hi_v):
            union(("v", lo_v), ("v", hi_v))
    for pt in (p, q):
        if pt[0] == "e":
            lo_v, hi_v = pt[1]
            if inside(lo_v):
                union(pt, ("v", lo_v))
            if inside(hi_v):
                union(pt, ("v", hi_v))
    return find(p) == find(q)


class DistanceTable:
    """
    Precomputed connectivity table of a graph, so that d_f of two points costs O(H).

    T[a][{x, y}] = the smallest level b such that vertices x, y (both of height >= a) are
    connected using only vertices of height in [a, b].  Built by one upward sweep with
    union-find for every lower level a: O(H * n^2) in total.
    """

    def __init__(self, G):
        self.G = G
        self.levels = sorted(set(G.f.values()))
        self.idx = {v: i for i, v in enumerate(G.f)}
        self.T = {a: self._sweep(a) for a in self.levels}

    def _key(self, x, y):
        i, j = self.idx[x], self.idx[y]
        return (i, j) if i < j else (j, i)

    def _sweep(self, a):
        G = self.G
        by_level = {}
        for v, h in G.f.items():
            if h >= a:
                by_level.setdefault(h, []).append(v)
        parent, members, out = {}, {}, {}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for b in sorted(by_level):
            for v in by_level[b]:
                parent[v], members[v] = v, [v]
            for v in by_level[b]:
                for lo_v, hi_v in G.incident_edges(v):
                    if hi_v != v or lo_v not in parent:  # only edges down to a vertex already present
                        continue
                    ra, rb = find(v), find(lo_v)
                    if ra == rb:
                        continue
                    if len(members[ra]) < len(members[rb]):
                        ra, rb = rb, ra
                    for x in members[ra]:
                        for y in members[rb]:
                            out[self._key(x, y)] = b
                    parent[rb] = ra
                    members[ra] += members.pop(rb)
        return out

    def distance(self, p, q):
        """d_f(p, q) for two normalized points."""
        G = self.G
        if p == q:
            return Fraction(0)
        hp, hq = G.height(p), G.height(q)
        lo, hi = min(hp, hq), max(hp, hq)
        if G.common_edge(p, q) is not None:
            return hi - lo
        ends = lambda r: [r[1]] if r[0] == "v" else list(r[1])
        # the vertices each point attaches to, with their (integer) heights
        pairs = [(x, y, G.f[x], G.f[y]) for x in ends(p) for y in ends(q)]
        # lower ends a: lo itself, then every level below lo, highest first
        k = bisect_left(self.levels, lo)
        lows = chain([lo], reversed(self.levels[:k]))
        best = None
        for a in lows:
            if best is not None and hi - a >= best:
                break  # every candidate from here on costs at least hi - a
            table = None
            for x, y, fx, fy in pairs:
                if a > fx or a > fy:
                    continue
                if x == y:
                    tb = fx
                else:
                    if table is None:
                        table = self.T[self.levels[bisect_left(self.levels, math.ceil(a))]]
                    tb = table.get(self._key(x, y))
                    if tb is None:
                        continue
                b = max(fx, fy, tb)  # integers
                if hi > b:
                    b = hi
                cand = b - a
                if best is None or cand < best:
                    best = cand
        if best is None:
            raise ValueError(f"{p} and {q} are in different components of the graph")
        return best


def graph_distance(G, p, q):
    """d_f(p, q) for two points of G (fast: builds the DistanceTable of G once and keeps it)."""
    table = getattr(G, "_distance_table", None)
    if table is None:
        table = G._distance_table = DistanceTable(G)
    return table.distance(G.normalize(p), G.normalize(q))


# ---------------------------------------------------------------------------
# height terms
# ---------------------------------------------------------------------------
def height_error(Phi):
    """Return the largest absolute height change under a map in either direction.

    Parameters:
        Phi (dict):
            Map data returned by ``FunctionalDistortionGraph.map_to``.

    Returns:
        tuple:
            The error in grid units and the (source point, image point)
            attaining it. On each piece the height difference is affine,
            so evaluating its endpoints suffices.
    """
    S, T = Phi["source"], Phi["target"]
    best, wit = Fraction(0), None

    def consider(sp, ip):
        nonlocal best, wit
        err = abs(S.height(sp) - T.height(ip))
        if wit is None or err > best:
            best, wit = err, (sp, ip)

    for v, ip in Phi["vertex_image"].items():
        consider(("v", v), ip)
    for e, segs in Phi["segments"].items():
        for s0, s1, p0, p1 in segs:
            for s, ip in ((s0, p0), (s1, p1)):
                consider(S.normalize(("e", e, S.f[e[0]] + s)), ip)
    return best, wit


# ---------------------------------------------------------------------------
# C-segments
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Side:
    """One coordinate of a C-segment: a point moving on `edge` with height h0 -> h1
    (linear in sigma), or a fixed vertex when edge is None."""
    edge: Optional[Any]
    vertex: Optional[Any]
    h0: Fraction
    h1: Fraction


@dataclass(frozen=True)
class CSegment:
    origin: str   # 'Phi' (a piece of an edge of F) or 'Psi' (a piece of an edge of G)
    edge: Any     # the edge of the source graph carrying the piece
    index: int    # index of the piece in segments[edge]
    F: Side       # the F-point (x(sigma))
    G: Side       # the G-point (y(sigma))


def side_point(graph, side, sigma):
    """The (normalized) point of `graph` that `side` is at when the parameter is sigma."""
    if side.edge is None:
        return ("v", side.vertex)
    h = side.h0 + sigma * (side.h1 - side.h0)
    return graph.normalize(("e", side.edge, h))


def _source_side(S, e, s0, s1):
    f0 = S.f[e[0]]
    return Side(e, None, f0 + s0, f0 + s1)


def _image_side(T, p0, p1):
    if p0 == p1:
        h = T.height(p0)
        if p0[0] == "v":
            return Side(None, p0[1], h, h)
        return Side(p0[1], None, h, h)
    return Side(T.common_edge(p0, p1), None, T.height(p0), T.height(p1))


def c_segments(Phi, Psi):
    """Return the pieces of the correspondence induced by both maps.

    A connected graph with one vertex contributes a constant piece for its
    vertex image. Its segment has ``edge=None`` and ``index=0``.
    """
    out = []
    for origin, M in (("Phi", Phi), ("Psi", Psi)):
        if not M["segments"]:
            for vertex, image in M["vertex_image"].items():
                src = _image_side(M["source"], ("v", vertex), ("v", vertex))
                img = _image_side(M["target"], image, image)
                sides = (src, img) if origin == "Phi" else (img, src)
                out.append(CSegment(origin, None, 0, *sides))
        for e in M["source"].edges():
            for i, (s0, s1, p0, p1) in enumerate(M["segments"][e]):
                src = _source_side(M["source"], e, s0, s1)
                img = _image_side(M["target"], p0, p1)
                if origin == "Phi":
                    out.append(CSegment(origin, e, i, src, img))
                else:
                    out.append(CSegment(origin, e, i, img, src))
    return out


# ---------------------------------------------------------------------------
# candidate points of one square
# ---------------------------------------------------------------------------
def _corners():
    return [(Fraction(s), Fraction(t)) for s in (0, 1) for t in (0, 1)]


def _lines(A, B):
    """Lines  a*sigma + b*tau = r  where one of the four expressions is a whole number."""
    lines = []
    for X, Y in ((A.F, B.F), (A.G, B.G)):
        dX, dY = X.h1 - X.h0, Y.h1 - Y.h0
        for c, a, b in ((X.h0 - Y.h0, dX, -dY),   # hX(sigma) - hY(tau)
                        (X.h0 + Y.h0, dX, dY)):   # hX(sigma) + hY(tau)
            if a == 0 and b == 0:
                continue  # constant: no line
            lo = c + min(0, a) + min(0, b)
            hi = c + max(0, a) + max(0, b)
            for k in range(math.floor(lo) + 1, math.ceil(hi)):  # integers with lo < k < hi
                lines.append((a, b, Fraction(k) - c))
    return lines


def candidate_points(A, B):
    """Return the points needed to maximize the discrepancy on one square.

    Parameters:
        A, B (CSegment):
            Two pieces of the correspondence. Their parameters vary
            independently from 0 to 1.

    Returns:
        list of tuple:
            Up to 18 distinct parameter pairs. These are the square corners,
            boundary intersections, and intersections of the height sum and
            difference lines from both graphs.

    For one graph, fix the edges containing the two points and write their
    ordered heights as x <= y. A route through vertices of minimum and maximum
    heights a and b has span max(y, b) - min(x, a). The possible vertex routes
    stay fixed in the edge interiors. With integer vertex heights and unit
    edges, each route's span is affine and has the form k, k - x, k + y, or
    y - x for integer k. A direct route on the same edge has span y - x.

    The cheapest formula can change only at integer heights or where x + y
    or x - y is an integer. Individual heights reach integers at the square
    boundary or are constant. The remaining lines partition the square into
    regions where both distances are affine. By continuity their absolute
    difference reaches its maximum at a corner of one of those regions.
    """
    pts = set(_corners())
    lines = _lines(A, B)
    inside = lambda x: 0 <= x <= 1
    for a, b, r in lines:
        for s0 in (0, 1):
            if b != 0 and inside(t := (r - a * s0) / b):
                pts.add((Fraction(s0), t))
        for t0 in (0, 1):
            if a != 0 and inside(s := (r - b * t0) / a):
                pts.add((s, Fraction(t0)))
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            a1, b1, r1 = lines[i]
            a2, b2, r2 = lines[j]
            det = a1 * b2 - a2 * b1
            if det == 0:
                continue
            s, t = (r1 * b2 - r2 * b1) / det, (a1 * r2 - a2 * r1) / det
            if inside(s) and inside(t):
                pts.add((s, t))
    return sorted(pts)


def grid_points(m):
    """(m+1) x (m+1) grid on the unit square (validation only)."""
    return [(Fraction(i, m), Fraction(j, m)) for i in range(m + 1) for j in range(m + 1)]


# ---------------------------------------------------------------------------
# D
# ---------------------------------------------------------------------------
def _square_max(F, G, A, B, cands):
    """Largest |d_f - d_g| over the candidate points of the square (A, B), with witness."""
    best, wit = None, None
    for s, t in cands:
        x, x2 = side_point(F, A.F, s), side_point(F, B.F, t)
        y, y2 = side_point(G, A.G, s), side_point(G, B.G, t)
        df, dg = graph_distance(F, x, x2), graph_distance(G, y, y2)
        gap = abs(df - dg)
        if best is None or gap > best:
            best = gap
            wit = dict(segA=A, segB=B, sigma=s, tau=t, pair=((x, y), (x2, y2)), d_f=df, d_g=dg)
    return best, wit


def _D(F, G, Phi, Psi, cand_fn):
    _check_maps(F, G, Phi, Psi)
    segs = c_segments(Phi, Psi)
    best, wit = Fraction(0), None
    for i in range(len(segs)):
        for j in range(i, len(segs)):
            gap, w = _square_max(F, G, segs[i], segs[j], cand_fn(segs[i], segs[j]))
            if wit is None or gap > best:
                best, wit = gap, w
    return best / 2, wit


def _check_maps(F, G, Phi, Psi):
    if Phi["source"] is not F or Phi["target"] is not G or Psi["source"] is not G or Psi["target"] is not F:
        raise ValueError("need Phi: F -> G and Psi: G -> F built on these very graph objects")


def distortion_D(F, G, Phi, Psi):
    """Exact D(Phi, Psi) (units of delta) and a witness: the square, (sigma, tau), the two
    C-points and the two distances where the maximum is attained."""
    return _D(F, G, Phi, Psi, candidate_points)


def D_breakpoints_only(F, G, Phi, Psi):
    """D from the corners of every square only (vertices and breakpoints). Can be too small."""
    return _D(F, G, Phi, Psi, lambda A, B: _corners())


def D_grid(F, G, Phi, Psi, m):
    """D from an (m+1) x (m+1) grid on every square. For validation only; never above the exact D."""
    pts = grid_points(m)
    return _D(F, G, Phi, Psi, lambda A, B: pts)


def check_grid_below_exact(F, G, Phi, Psi, m=4, every=1):
    """Assert, square by square, that the grid maximum does not exceed the exact one.
    `every=k` checks only every k-th piece in each direction (to keep large examples quick).
    Returns the number of squares checked."""
    _check_maps(F, G, Phi, Psi)
    segs = c_segments(Phi, Psi)
    pts = grid_points(m)
    n = 0
    for i in range(0, len(segs), every):
        for j in range(i, len(segs), every):
            exact, _ = _square_max(F, G, segs[i], segs[j], candidate_points(segs[i], segs[j]))
            grid, w = _square_max(F, G, segs[i], segs[j], pts)
            assert grid <= exact, f"grid beats exact on square {segs[i]} x {segs[j]}: {grid} > {exact} at {w}"
            n += 1
    return n


# ---------------------------------------------------------------------------
# round-trip errors and pruning
# ---------------------------------------------------------------------------
def _line_candidates(X, Y):
    """
    Two points of ONE graph moving with a common parameter rho in [0, 1] (Sides X and Y).
    Their distance is piecewise linear in rho and bends only where  hX - hY  or  hX + hY  is a
    whole number, so it is largest at rho = 0, 1 or at such a point.  Returns those rho, sorted.
    """
    pts = {Fraction(0), Fraction(1)}
    dX, dY = X.h1 - X.h0, Y.h1 - Y.h0
    for c, a in ((X.h0 - Y.h0, dX - dY), (X.h0 + Y.h0, dX + dY)):
        if a == 0:
            continue
        lo, hi = min(c, c + a), max(c, c + a)
        for k in range(math.floor(lo) + 1, math.ceil(hi)):  # integers with lo < k < hi
            pts.add((Fraction(k) - c) / a)
    return sorted(pts)


def _roundtrip_parts(M, N, e, piece, starts):
    """
    One piece of M: S -> T (on the source edge e) is cut where its image passes a breakpoint of
    N: T -> S.  On each part both x and N(M(x)) move linearly on fixed edges of S.
    Yields (X, Y): the Sides of S for x and for N(M(x)), with a common parameter.
    `starts[E]` = the list of piece start parameters of N on the target edge E.
    """
    S, T = M["source"], M["target"]
    s0, s1, p0, p1 = piece
    f0 = S.f[e[0]]
    if p0 == p1:                                   # the whole piece goes to one point
        z = evaluate_map(N, p0)
        yield Side(e, None, f0 + s0, f0 + s1), _image_side(S, z, z)
        return
    E_ = T.common_edge(p0, p1)
    g0 = T.f[E_[0]]
    ua, ub = T.height(p0) - g0, T.height(p1) - g0  # the image runs over [ua, ub] on E_ (ua != ub)
    lo, hi = min(ua, ub), max(ua, ub)
    segs = N["segments"][E_]
    first = max(bisect_right(starts[E_], lo) - 1, 0)
    for u0, u1, q0, q1 in segs[first:]:
        if u0 >= hi:
            break
        a, b = max(u0, lo), min(u1, hi)
        if a >= b:
            continue
        # the point x whose image sits at parameter u of E_
        xa = f0 + s0 + (a - ua) / (ub - ua) * (s1 - s0)
        xb = f0 + s0 + (b - ua) / (ub - ua) * (s1 - s0)
        if q0 == q1:
            Y = _image_side(S, q0, q0)
        else:
            h0, h1 = S.height(q0), S.height(q1)
            Y = Side(S.common_edge(q0, q1), None,
                     h0 + (a - u0) / (u1 - u0) * (h1 - h0),
                     h0 + (b - u0) / (u1 - u0) * (h1 - h0))
        yield Side(e, None, xa, xb), Y


def roundtrip_errors(Phi, Psi):
    """
    The round-trip error of every C-segment, exact, in the order of c_segments(Phi, Psi):
      for a piece of Phi (on F):  max over the piece of  d_f(x, Psi(Phi(x)))
      for a piece of Psi (on G):  max over the piece of  d_g(y, Phi(Psi(y)))
    """
    out = []
    for M, N in ((Phi, Psi), (Psi, Phi)):
        S = M["source"]
        starts = {E_: [seg[0] for seg in segs] for E_, segs in N["segments"].items()}
        if not M["segments"]:
            out.extend(graph_distance(S, ("v", v), evaluate_map(N, image))
                       for v, image in M["vertex_image"].items())
        for e in S.edges():
            for piece in M["segments"][e]:
                best = Fraction(0)
                for X, Y in _roundtrip_parts(M, N, e, piece, starts):
                    for rho in _line_candidates(X, Y):
                        d = graph_distance(S, side_point(S, X, rho), side_point(S, Y, rho))
                        if d > best:
                            best = d
                out.append(best)
    return out


def pair_bound(A, B, rA, rB, eps_Phi, eps_Psi):
    """
    U(A, B): an upper bound for |d_f - d_g| on the square of the C-segments A, B, from their
    round-trip errors rA, rB and the two height errors only (no distance between A and B):
        A, B from Phi:           max(2 eps_Phi,  rA + rB + 2 eps_Psi)
        A from Phi, B from Psi:  max(2 eps_Phi + rB,  rA + 2 eps_Psi)
        A, B from Psi:           max(2 eps_Psi,  rA + rB + 2 eps_Phi)
    """
    if A.origin == B.origin:
        own, other = (eps_Phi, eps_Psi) if A.origin == "Phi" else (eps_Psi, eps_Phi)
        return max(2 * own, rA + rB + 2 * other)
    if A.origin == "Psi":
        rA, rB = rB, rA
    return max(2 * eps_Phi + rB, rA + 2 * eps_Psi)


def distortion_D_pruned(F, G, Phi, Psi, floor=0):
    """
    D(Phi, Psi) without looking at every square.  Returns (D, witness, stats).

    A square (A, B) is skipped when pair_bound(A, B) <= max(largest |d_f - d_g| found so far, floor).
    The segments of each map are sorted by round-trip error, so for a fixed A the squares that
    must be looked at come first and the skipped ones are never listed.

    floor = 0 (default): the returned D is exact, always (same value as distortion_D).
    floor = 2 * max(height errors): squares that cannot raise FD above the height errors are
        skipped as well.  max(D returned, height errors) is still the exact FD, but the returned
        D itself is exact only if stats["D_exact"] (it is then at least floor / 2); otherwise it
        is a lower bound of D and D <= floor / 2.

    stats: pairs_total, pairs_visited, D_exact, roundtrip (the list from roundtrip_errors).
    The witness belongs to the returned value (a different square than distortion_D may be
    reported when several attain the maximum).
    """
    _check_maps(F, G, Phi, Psi)
    floor = Fraction(floor)
    segs = c_segments(Phi, Psi)
    rb = roundtrip_errors(Phi, Psi)
    eP, eQ = height_error(Phi)[0], height_error(Psi)[0]
    # indices of the segments of each map, largest round-trip error first (ties: original order)
    iP = sorted((i for i, s in enumerate(segs) if s.origin == "Phi"), key=lambda i: -rb[i])
    iQ = sorted((i for i, s in enumerate(segs) if s.origin == "Psi"), key=lambda i: -rb[i])
    best, witness, visited = Fraction(0), None, 0

    def cutoff():
        return max(best, floor)

    def visit(i, j):
        nonlocal best, witness, visited
        if i > j:
            i, j = j, i
        gap, w = _square_max(F, G, segs[i], segs[j], candidate_points(segs[i], segs[j]))
        visited += 1
        if witness is None or gap > best:
            best, witness = gap, w

    # two segments of the same map: U = max(2 own, r + r' + 2 other), decreasing along the sorted list
    for idx, own, other in ((iP, 2 * eP, 2 * eQ), (iQ, 2 * eQ, 2 * eP)):
        for a in range(len(idx)):
            if max(own, 2 * rb[idx[a]] + other) <= cutoff():
                break
            for b in range(a, len(idx)):
                if max(own, rb[idx[a]] + rb[idx[b]] + other) <= cutoff():
                    break
                visit(idx[a], idx[b])
    # one segment of each map: skipped iff  r(A) + 2 eps_Psi <= T  and  r'(B) + 2 eps_Phi <= T
    done = 0
    for i in iP:
        if rb[i] + 2 * eQ <= cutoff():
            break
        done += 1
        for j in iQ:
            visit(i, j)
    for j in iQ:
        if rb[j] + 2 * eP <= cutoff():
            break
        for i in iP[done:]:
            visit(i, j)
    if witness is None and segs:      # nothing had to be looked at: still report a witness
        visit(0, 0)
    P = len(segs)
    stats = {"pairs_total": P * (P + 1) // 2, "pairs_visited": visited,
             "D_exact": best >= floor, "roundtrip": rb}
    return best / 2, witness, stats


# ---------------------------------------------------------------------------
# everything together
# ---------------------------------------------------------------------------
def functional_distortion(F, G, Phi, Psi, prune=True, exact_D=True):
    """Evaluate functional distortion for a fixed pair of maps.

    Parameters:
        F, G (FunctionalDistortionGraph):
            Connected graphs with the same grid spacing, ``delta``.
        Phi, Psi (dict):
            Maps from F to G and G to F, built on these graph objects.
        prune (bool, optional):
            Skip pairs of pieces whose upper bound cannot improve the
            result. Defaults to True.
        exact_D (bool, optional):
            With pruning, require the distance distortion term D itself
            to be exact. Defaults to True. If False, the height errors can
            serve as a cutoff and the final FD is still exact.

    Returns:
        dict:
            Both height errors, ``D``, and ``FD``, the maximum of those three
            terms. Keys ending in ``_units`` give exact Fractions in grid
            units. The corresponding plain keys are scaled by delta.
            Witnesses give points attaining the reported values.
            ``pairs_visited`` and ``pairs_total`` count piece pairs.
            If ``D_exact`` is false, the reported D is a lower bound.

    Examples:
        >>> from examples.graph_examples import square_loop_example
        >>> result = functional_distortion(*square_loop_example())
        >>> result["D_units"], result["FD_units"]
        (Fraction(3, 4), Fraction(1, 1))
    """
    _check_maps(F, G, Phi, Psi)
    if F.delta != G.delta:
        raise ValueError(f"F.delta = {F.delta} but G.delta = {G.delta}")
    hPhi, wPhi = height_error(Phi)
    hPsi, wPsi = height_error(Psi)
    if prune:
        D, wD, stats = distortion_D_pruned(F, G, Phi, Psi, floor=0 if exact_D else 2 * max(hPhi, hPsi))
    else:
        D, wD = distortion_D(F, G, Phi, Psi)
        P = len(c_segments(Phi, Psi))
        stats = {"pairs_total": P * (P + 1) // 2, "pairs_visited": P * (P + 1) // 2, "D_exact": True}
    FD = max(hPhi, hPsi, D)
    d = F.delta
    return {
        "height_Phi_units": hPhi, "height_Psi_units": hPsi, "D_units": D, "FD_units": FD,
        "height_Phi": hPhi * d, "height_Psi": hPsi * d, "D": D * d, "FD": FD * d,
        "witness_height_Phi": wPhi, "witness_height_Psi": wPsi, "witness_D": wD,
        "delta": d,
        "pairs_total": stats["pairs_total"], "pairs_visited": stats["pairs_visited"],
        "D_exact": stats["D_exact"],
    }
