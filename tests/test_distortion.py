"""Tests for functionaldistortiongraph.py and distortion.py.  Run:  python3 -m pytest -q   (the tests themselves need no pytest, so the notebook can call them)"""
from contextlib import contextmanager
from fractions import Fraction as Fr

from core.distortion import D_breakpoints_only, functional_distortion
from core.functionaldistortiongraph import E, FunctionalDistortionGraph, V, evaluate_map


def _square_loop_example():
    """Example 2: the square loop v-a-w-b-v folded onto one edge cd."""
    F = FunctionalDistortionGraph.from_edges({"v": 0, "a": 1, "b": 1, "w": 2}, [("v", "a"), ("v", "b"), ("a", "w"), ("b", "w")])
    G = FunctionalDistortionGraph.from_edges({"c": 0, "d": 1}, [("c", "d")])
    Phi = F.map_to(G, {"v": "c", "a": "d", "b": "c", "w": "d"})
    Psi = G.map_to(F, {"c": "v", "d": "a"})
    return F, G, Phi, Psi


def _torus_line_example(h=3):
    """A torus with a loop of height h (two arms between heights 1 and 1 + h) and a line, both
    covering the heights 0 .. h + 2.  Phi sends every point to the point of the line at the same
    height; Psi sends the line to the path through the right arm.  Both maps keep the height,
    D = FD = h/4, and vertices alone give floor(h/2)/2."""
    top = h + 2
    stem = {("s", t): t for t in (0, 1, top - 1, top)}
    arms = {(side, t): t for side in ("L", "R") for t in range(2, top - 1)}
    edges = [(("s", 0), ("s", 1)), (("s", top - 1), ("s", top))]
    for side in ("L", "R"):
        chain = [("s", 1)] + [(side, t) for t in range(2, top - 1)] + [("s", top - 1)]
        edges += list(zip(chain, chain[1:]))
    F = FunctionalDistortionGraph.from_edges({**stem, **arms}, edges)
    G = FunctionalDistortionGraph.from_edges({t: t for t in range(top + 1)}, [(t, t + 1) for t in range(top)])
    Phi = F.map_to(G, {v: t for v, t in F.f.items()})
    Psi = G.map_to(F, {t: (("R", t) if 2 <= t <= top - 2 else ("s", t)) for t in G.f})
    return F, G, Phi, Psi


@contextmanager
def raises(exc):
    """Stand-in for pytest.raises, so that these tests (and the notebook) do not need pytest."""
    try:
        yield
    except exc:
        return
    raise AssertionError(f"{exc.__name__} was not raised")


# ---- Example 1 -----------------------------------------------------------
# ---- Example 2 -----------------------------------------------------------
def test_square_loop_values():
    F, G, Phi, Psi = _square_loop_example()
    r = functional_distortion(F, G, Phi, Psi)
    assert r["height_Phi_units"] == 1
    assert r["height_Psi_units"] == 0
    assert r["D_units"] == Fr(3, 4)
    assert r["FD_units"] == 1
    assert D_breakpoints_only(F, G, Phi, Psi)[0] == Fr(1, 2)
    # witness of D: the midpoints of va and bw, d_f = 3/2, d_g = 0
    w = r["witness_D"]
    assert (w["d_f"], w["d_g"]) == (Fr(3, 2), 0)
    assert {w["pair"][0][0], w["pair"][1][0]} == {("e", ("v", "a"), Fr(1, 2)), ("e", ("b", "w"), Fr(3, 2))}


# ---- Piecewise-linear maps ---------------------------------------------------------------
def test_three_edge_route_breakpoints():
    F = FunctionalDistortionGraph.from_edges({"u": 0, "v": 1}, [("u", "v")])
    G = FunctionalDistortionGraph.from_edges({i: i for i in range(4)}, [(0, 1), (1, 2), (2, 3)])
    Phi = F.map_to(G, {"u": 0, "v": 3}, routes={("u", "v"): [1, 2]})
    segs = Phi["segments"][("u", "v")]
    assert [(s0, s1) for s0, s1, _, _ in segs] == [(0, Fr(1, 3)), (Fr(1, 3), Fr(2, 3)), (Fr(2, 3), 1)]
    assert evaluate_map(Phi, E(("u", "v"), Fr(1, 2))) == ("e", (1, 2), Fr(3, 2))  # halfway = middle of edge 1-2
    # unevenly spaced breakpoints
    Phi2 = F.map_to(G, {"u": 0, "v": 3}, routes={("u", "v"): [1, 2]},
                             reparameterizations={("u", "v"): [Fr(1, 10), Fr(1, 2)]})
    assert Phi2["segments"][("u", "v")][1][:2] == (Fr(1, 10), Fr(1, 2))


def test_check_rejects_bad_maps():
    F = FunctionalDistortionGraph.from_edges({"u": 0, "v": 1}, [("u", "v")])
    G = FunctionalDistortionGraph.from_edges({i: i for i in range(4)}, [(0, 1), (1, 2), (2, 3)])
    with raises(ValueError):                      # 0 and 2 are not on one edge
        F.map_to(G, {"u": 0, "v": 3}, routes={("u", "v"): [2]})
    with raises(ValueError):                      # reparameterization not increasing
        F.map_to(G, {"u": 0, "v": 3}, routes={("u", "v"): [1, 2]},
                          reparameterizations={("u", "v"): [Fr(1, 2), Fr(1, 3)]})
    with raises(TypeError):
        F.map_to(G, {"u": 0, "v": 3}, routes={("u", "v"): [1, 2]},
                          reparameterizations={("u", "v"): [0.3, 0.6]})


# ---- round-trip errors and pruning ----------------------------------------
def test_pruning_preserves_distortion():
    graphs_and_maps = _torus_line_example(h=3)
    pruned = functional_distortion(*graphs_and_maps)
    exhaustive = functional_distortion(*graphs_and_maps, prune=False)
    assert pruned["D_units"] == exhaustive["D_units"] == Fr(3, 4)
    assert pruned["FD_units"] == exhaustive["FD_units"] == Fr(3, 4)
    assert pruned["pairs_visited"] < exhaustive["pairs_visited"]


def test_distortion_with_single_vertex_graphs():
    F = FunctionalDistortionGraph.from_edges({"u": 2}, [])
    G = FunctionalDistortionGraph.from_edges({"v": 3}, [])
    phi = F.map_to(G, {"u": "v"})
    psi = G.map_to(F, {"v": "u"})
    pair = (V("u"), V("v"))

    for prune in (True, False):
        result = functional_distortion(F, G, phi, psi, prune=prune)
        assert result["D_units"] == 0
        assert result["FD_units"] == 1
        assert result["witness_D"]["pair"] == (pair, pair)


    F = FunctionalDistortionGraph.from_edges({"u": 0}, [])
    G = FunctionalDistortionGraph.from_edges({"a": 0, "b": 1}, [("a", "b")])
    phi = F.map_to(G, {"u": E(("a", "b"), "1/2")})
    psi = G.map_to(F, {"a": "u", "b": "u"})

    for prune in (True, False):
        result = functional_distortion(F, G, phi, psi, prune=prune)
        assert result["D_units"] == Fr(1, 2)
        assert result["FD_units"] == 1
        assert result["witness_D"]["d_g"] == 1

if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
