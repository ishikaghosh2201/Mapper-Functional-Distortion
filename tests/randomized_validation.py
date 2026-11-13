"""Optional randomized checks. Run with pytest tests/randomized_validation.py."""

import random
from fractions import Fraction as Fr

from core.distortion import (
    _square_max,
    c_segments,
    candidate_points,
    distortion_D,
    distortion_D_pruned,
    graph_distance,
    height_error,
    pair_bound,
    roundtrip_errors,
    side_point,
)
from core.fd_map_build import ETA, frechet_times
from core.functionaldistortiongraph import E, V, evaluate_map


def test_fast_distance_matches_simple():
    """The table-based d_f equals the simple union-find d_f on random graphs and random points."""
    from core.distortion import graph_distance_simple
    from tests.random_check import random_graph
    rng = random.Random(1)
    checked = 0
    for _ in range(40):
        G = random_graph(rng, nlevels=rng.choice([3, 4, 5]), maxw=3)

        def rand_point():
            if rng.random() < 0.3:
                return V(rng.choice(list(G.f)))
            e = rng.choice(list(G.edges()))
            return E(e, G.f[e[0]] + Fr(rng.randint(1, 5), 6))
        for _ in range(60):
            p, q = rand_point(), rand_point()
            assert graph_distance(G, p, q) == graph_distance_simple(G, p, q), (p, q)
            checked += 1
    assert checked == 2400


def test_pruned_equals_exhaustive_on_random_maps():
    """On random graphs and maps: the round-trip error of a piece is not beaten by sampled points,
    the bound U holds on every square, and both pruning modes give the exhaustive values."""
    from tests.random_check import random_graph, random_map
    rng = random.Random(5)
    for _ in range(8):
        F, G = random_graph(rng), random_graph(rng)
        Phi, Psi = random_map(rng, F, G), random_map(rng, G, F)
        segs, rb = c_segments(Phi, Psi), roundtrip_errors(Phi, Psi)
        eP, eQ = height_error(Phi)[0], height_error(Psi)[0]
        for s, r in zip(segs, rb):
            M, N, S, side = (Phi, Psi, F, s.F) if s.origin == "Phi" else (Psi, Phi, G, s.G)
            for k in range(13):
                x = side_point(S, side, Fr(k, 12))
                assert graph_distance(S, x, evaluate_map(N, evaluate_map(M, x))) <= r
        for i in range(len(segs)):
            for j in range(i, len(segs)):
                gap, _ = _square_max(F, G, segs[i], segs[j], candidate_points(segs[i], segs[j]))
                assert gap <= pair_bound(segs[i], segs[j], rb[i], rb[j], eP, eQ)
        D = distortion_D(F, G, Phi, Psi)[0]
        D1, w1, st1 = distortion_D_pruned(F, G, Phi, Psi)
        assert D1 == D and st1["D_exact"] and abs(w1["d_f"] - w1["d_g"]) == 2 * D
        D2, _, st2 = distortion_D_pruned(F, G, Phi, Psi, floor=2 * max(eP, eQ))
        assert max(D2, eP, eQ) == max(D, eP, eQ)          # FD is exact in both modes
        assert D2 == D if st2["D_exact"] else D2 <= D <= max(eP, eQ)


def _feasible(t, h, eps):
    """Free-space decision: is there s_0 = 0 <= s_1 <= ... <= s_k = 1 with |t + s_i - h_i| <= eps?
    (greedy: go as early as each window allows)"""
    t, h = Fr(t), [Fr(x) for x in h]
    lo = [max(Fr(0), x - t - eps) for x in h]
    hi = [min(Fr(1), x - t + eps) for x in h]
    if not (lo[0] <= 0 <= hi[0] and lo[-1] <= 1 <= hi[-1]):
        return False
    s = Fr(0)
    for i in range(1, len(h) - 1):
        s = max(s, lo[i])
        if s > hi[i]:
            return False
    return s <= 1


def test_frechet_optimum_and_strict_parameters():
    """Check the ideal tolerance and the realized parameters against free space."""
    rng = random.Random(7)
    for _ in range(400):
        t = rng.randint(0, 3)
        k = rng.randint(1, 10)
        h = [t + rng.choice([-1, 0, 1, 2])]
        for _ in range(k):
            h.append(h[-1] + rng.choice([-1, 1]))
        times, d, _ = frechet_times(t, h)
        assert len(times) == len(h) and times[0] == 0 and times[-1] == 1
        assert all(a < b for a, b in zip(times, times[1:]))
        assert all(isinstance(time, Fr) for time in times)
        assert max(abs(t + time - height) for time, height in zip(times, h)) <= d + ETA
        assert _feasible(t, h, d), (t, h, d)
        if d > 0:
            assert not _feasible(t, h, d - Fr(1, 10 ** 6)), (t, h, d)
