"""Tests for the Frechet reparameterization in fd_map_build.py.  Run:  python -m pytest -q tests/test_map_recovery.py"""
import sys
from fractions import Fraction as Fr

import pytest

from core.fd_map_build import ETA, frechet_times
from core.functionaldistortiongraph import FunctionalDistortionGraph, evaluate_map


def test_frechet_ties_and_routes_leaving_the_height_band():
    # heights 0,1,0,1 on the edge [0,1]: half the largest drop is 1/2, vertices 1 and 2 both want s = 1/2;
    # they are moved to 1/2 -+ ETA/6, and the error becomes 1/2 + ETA/6
    times, d, spread = frechet_times(0, [0, 1, 0, 1])
    assert d == Fr(1, 2) and spread
    assert times == [0, Fr(1, 2) - ETA / 6, Fr(1, 2) + ETA / 6, 1]
    assert max(abs(s - h) for s, h in zip(times, [0, 1, 0, 1])) == Fr(1, 2) + ETA / 6
    assert all(isinstance(s, Fr) for s in times)

    # edge [0,1], route climbs to height 3 and comes back to 1: the peak is 2 above the edge, so d_e = 2
    # (a formula that sees the edge as a whole line gives 1)
    times, d, spread = frechet_times(0, [0, 1, 2, 3, 2, 1])
    assert d == 2 and spread
    assert max(abs(s - h) for s, h in zip(times, [0, 1, 2, 3, 2, 1])) <= d + ETA
    # edge [10,11], heights 10,11,12,11,12,11: 12 is 1 above the edge
    assert frechet_times(10, [10, 11, 12, 11, 12, 11])[1] == 1

def test_torus_to_line():
    """F = torus(0,1,4,5), G = line(0,5) (n = 1): height errors 0 and 0, D = FD = 3/4; breakpoints only 1/2."""
    pytest.importorskip("cereeberus")
    from cereeberus import Interleave
    from cereeberus.data import ex_mappergraphs as exm, ex_reebgraphs as exr
    from core.distortion import D_breakpoints_only, functional_distortion
    from core.fd_map_build import failing_handoffs, maps_from_interleave

    I = Interleave(exm.torus(0, 1, 4, 5, delta=0.1), exr.line(0, 5).to_mapper(delta=0.1))
    assert I.fit() == 1
    m = maps_from_interleave(I, seed=0, points="close", paths="min_span")
    F, G, Phi, Psi = m["F"], m["G"], m["Phi"], m["Psi"]
    r = functional_distortion(F, G, Phi, Psi)
    assert (r["height_Phi_units"], r["height_Psi_units"], r["D_units"], r["FD_units"]) == (0, 0, Fr(3, 4), Fr(3, 4))
    assert D_breakpoints_only(F, G, Phi, Psi)[0] == Fr(1, 2)
    assert not failing_handoffs(Phi) and not failing_handoffs(Psi)


def test_maps_from_a_zero_interleaving():
    """Isomorphic graphs: fit() returns n = 0, and the maps are the isomorphisms, with FD = 0."""
    pytest.importorskip("cereeberus")
    from cereeberus import Interleave, MapperGraph
    from core.distortion import functional_distortion
    from core.fd_map_build import failing_handoffs, maps_from_interleave

    def graph(nodes, edges):
        M = MapperGraph()
        for name, h in nodes:
            M.add_node(name, h, reset_pos=False)
        for u, v in edges:
            M.add_edge(u, v, reset_pos=False)
        return M

    nodes = [("v", 0), ("a", 1), ("b", 1), ("w", 2), ("t", 3)]
    edges = [("v", "a"), ("v", "b"), ("a", "w"), ("b", "w"), ("w", "t")]
    relabelled = graph([(x.upper(), h) for x, h in nodes], [(x.upper(), y.upper()) for x, y in edges])
    I = Interleave(graph(nodes, edges), relabelled)
    assert I.fit() == 0
    m = maps_from_interleave(I, seed=0, points="close", paths="min_span")
    F, G, Phi, Psi = m["F"], m["G"], m["Phi"], m["Psi"]
    assert m["n"] == 0
    assert all(evaluate_map(Psi, evaluate_map(Phi, ("v", v))) == ("v", v) for v in F.f)
    assert all(evaluate_map(Phi, evaluate_map(Psi, ("v", y))) == ("v", y) for y in G.f)
    r = functional_distortion(F, G, Phi, Psi)
    assert (r["height_Phi_units"], r["height_Psi_units"], r["D_units"], r["FD_units"]) == (0, 0, 0, 0)
    assert functional_distortion(F, G, Phi, Psi, prune=False)["FD_units"] == 0
    assert not failing_handoffs(Phi) and not failing_handoffs(Psi)


def test_explicit_interleaving_without_solver():
    from cereeberus.distance.interleave import Assignment
    from core.fd_map_build import maps_from_explicit_interleaving

    graph = FunctionalDistortionGraph.from_edges({0: 0, 1: 1}, [(0, 1)]).mapper
    assignment = Assignment(graph, graph, n=1)

    def level_maps(source, target):
        return {
            kind: {obj: targets[height][0] for height, objects in sources.items() for obj in objects}
            for kind, sources, targets in (
                ("V", source.func_to_vertex_dict(), target.func_to_vertex_dict()),
                ("E", source.func_to_edge_dict(), target.func_to_edge_dict()),
            )
        }

    phi = {key: level_maps(assignment.F(key), assignment.G(target))
           for key, target in (("0", "n"), ("n", "2n"))}
    psi = {key: level_maps(assignment.G(key), assignment.F(target))
           for key, target in (("0", "n"), ("n", "2n"))}
    maps = maps_from_explicit_interleaving(assignment, phi, psi)
    assert maps["Phi"]["vertex_image"] == {0: ("v", 0), 1: ("v", 1)}
    assert not assignment.phi("0", "V")[0].array.any()

    phi["0"]["V"][0] = phi["0"]["V"][1]
    with pytest.raises(ValueError, match="needs an image at level 0"):
        maps_from_explicit_interleaving(assignment, phi, psi)


def test_explicit_interleaving_rejects_noncommuting_diagrams():
    from cereeberus.distance.interleave import Assignment
    from core.fd_map_build import maps_from_explicit_interleaving

    graph = FunctionalDistortionGraph.from_edges(
        {"u": 0, "a": 1, "b": 1, "v": 2},
        [("u", "a"), ("u", "b"), ("a", "v"), ("b", "v")],
    ).mapper
    assignment = Assignment(graph, graph, n=0)
    identity = {v: v for v in graph.nodes}
    swap = {**identity, "a": "b", "b": "a"}
    edges = list(graph.edges(keys=True))

    def map_at_level(vertices, edge_vertices):
        return {"V": vertices, "E": {(u, v, k): (edge_vertices[u], edge_vertices[v], k)
                                      for u, v, k in edges}}

    unchanged = map_at_level(identity, identity)
    swapped = map_at_level(swap, swap)
    phi = {"0": unchanged, "n": unchanged}
    psi = {"0": swapped, "n": swapped}
    cases = [
        ("edge incidence", {**phi, "0": map_at_level(identity, swap)}, psi),
        ("smoothing compatibility", {**phi, "n": swapped}, {**psi, "n": unchanged}),
        ("interleaving triangle", phi, psi),
    ]
    for violation, forward, reverse in cases:
        with pytest.raises(ValueError, match=violation):
            maps_from_explicit_interleaving(assignment, forward, reverse)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
