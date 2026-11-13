"""Random stress test: on random graphs and random PL maps, no grid point may beat the exact value, square by square."""
import random
from collections import deque
from fractions import Fraction as Fr

from core.distortion import check_grid_below_exact, distortion_D, D_breakpoints_only
from core.functionaldistortiongraph import FunctionalDistortionGraph


def random_graph(rng, nlevels=3, maxw=2):
    heights, levels = {}, []
    for t in range(nlevels):
        k = rng.randint(1, maxw)
        levels.append([(t, i) for i in range(k)])
        for v in levels[-1]:
            heights[v] = t
    edges = set()
    for t in range(nlevels - 1):  # make it connected: every vertex gets a neighbour on both sides
        for v in levels[t]:
            edges.add((v, rng.choice(levels[t + 1])))
        for w in levels[t + 1]:
            edges.add((rng.choice(levels[t]), w))
        for v in levels[t]:
            for w in levels[t + 1]:
                if rng.random() < 0.25:
                    edges.add((v, w))
    return FunctionalDistortionGraph.from_edges(heights, sorted(edges))


def bfs_path(G, a, b):
    prev, q = {a: None}, deque([a])
    while q:
        x = q.popleft()
        for lo, hi in G.incident_edges(x):
            y = hi if lo == x else lo
            if y not in prev:
                prev[y] = x
                q.append(y)
    path = [b]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    return path[::-1]


def neighbors(G, x):
    return [hi if lo == x else lo for lo, hi in G.incident_edges(x)]


def random_map(rng, S, T):
    vimg = {v: rng.choice(list(T.f)) for v in S.f}
    routes, reparameterizations = {}, {}
    for e in S.edges():
        path = bfs_path(T, vimg[e[0]], vimg[e[1]])
        route = path[1:-1]
        if rng.random() < 0.4:  # detour / dwell / backtrack
            i = rng.randint(0, len(route))
            cur = (path[:1] + route)[i]
            nb = rng.choice(neighbors(T, cur) + [cur])
            route = route[:i] + [nb, cur] + route[i:] if nb != cur else route[:i] + [cur] + route[i:]
            if i == 0 and route and route[0] == vimg[e[0]] and False:
                pass
        k = len(route)
        den = rng.choice([2, 3, 5, 7])
        if k < den:
            cuts = sorted(rng.sample(range(1, den), k))
            reparameterizations[e] = [Fr(c, den) for c in cuts]
        routes[e] = route
    return S.map_to(T, vimg, routes, reparameterizations)


def main(n=40, seed=0):
    rng = random.Random(seed)
    squares = bigger = 0
    for trial in range(n):
        F, G = random_graph(rng), random_graph(rng)
        Phi, Psi = random_map(rng, F, G), random_map(rng, G, F)
        m = rng.choice([4, 6])
        squares += check_grid_below_exact(F, G, Phi, Psi, m=m)
        if distortion_D(F, G, Phi, Psi)[0] > D_breakpoints_only(F, G, Phi, Psi)[0]:
            bigger += 1
    print(f"{n} random instances, {squares} squares: grid never beat exact; "
          f"exact D > breakpoints-only D in {bigger} instances")


if __name__ == "__main__":
    main()
