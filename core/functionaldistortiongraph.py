"""Continuous realizations of ceREEBerus Mapper graphs and maps between them.

An edge is identified by its endpoints ``(lower_vertex, upper_vertex)``.
Use ``V(v)`` for a vertex and ``E(edge, h)`` for a point on an edge at
height ``h``. Heights are in grid units, so an edge from level 2 to level 3
has its midpoint at height 5/2. Multiply by ``delta`` for the original units.

Use integers, ``Fraction`` objects, or strings such as ``"5/2"`` for exact
heights and map parameters. Floating-point inputs raise ``TypeError``.
"""
from fractions import Fraction
import networkx as nx
from cereeberus import MapperGraph


def _frac(x):
    if isinstance(x, float):
        raise TypeError(f"floats are not allowed (got {x!r}); use int, Fraction or str")
    return Fraction(x)


def V(v):
    """Return the tagged point ``("v", v)`` for vertex `v`."""
    return ("v", v)


def E(edge, h):
    """Return the tagged point ``("e", edge, h)``, with `h` as a Fraction.

    Parameters:
        edge : tuple
            Endpoints ordered from lower to higher height.
        h : int, Fraction, or str
            Height in grid units. Pass the result to the graph's
            `normalize` method to check the height and resolve endpoints.
    """
    return ("e", edge, _frac(h))


class FunctionalDistortionGraph:
    """Represent the continuous realization of an integer-grid MapperGraph.

    The underlying MapperGraph supplies the vertices, edges, and height
    function. Each edge represents the full interval between its endpoint
    heights, so maps can send points to vertices or to edge interiors.
    MapperGraph subdivides edges at intermediate grid levels. It allows
    parallel edges, but this class requires each endpoint pair to identify
    a unique edge.

    Parameters:
        mapper : MapperGraph
            Existing ceREEBerus MapperGraph without parallel edges. The
            instance keeps a reference to it. Keep its topology and heights
            fixed while using maps built from it.

    Attributes:
        f : dict
            Vertex heights in grid units, taken from `mapper.f`, the function values.
        delta : float
            Grid spacing in original height units, taken from `mapper.delta`.

    Raises:
        ValueError
            If the graph has parallel edges or violates the MapperGraph
            invariant that every edge runs uphill by one grid level.
    """

    def __init__(self, mapper):
        for u, v in mapper.edges():
            if mapper.f[v] - mapper.f[u] != 1:
                raise ValueError(f"edge {(u, v)!r} does not join heights t and t+1")
            if mapper.number_of_edges(u, v) > 1:
                raise ValueError(f"double edge {(u, v)!r}")
        self.mapper = mapper

    @classmethod
    def from_edges(cls, f, edges, delta=1):
        """Construct a continuous graph from vertex heights and edges.

        Parameters:
            f : dict
                Maps each vertex label to an integer grid level. Include
                isolated vertices here as well.
            edges : iterable of tuple
                Endpoint pairs in either order. Each pair must join
                consecutive grid levels and appear only once.
            delta : float, optional
                Grid spacing in original units. Defaults to 1.

        Returns:
            FunctionalDistortionGraph
                The continuous graph backed by a new ceREEBerus MapperGraph.
        """
        graph = nx.Graph()
        graph.add_nodes_from(f)
        for u, v in edges:
            if abs(f[u] - f[v]) != 1:
                raise ValueError(f"edge {(u, v)!r} does not join heights t and t+1")
            if graph.has_edge(u, v):
                raise ValueError(f"double edge {(u, v)!r}")
            graph.add_edge(u, v)
        return cls(MapperGraph(graph, dict(f), delta=delta))

    @property
    def f(self):
        """Vertex heights in grid units."""
        return self.mapper.f

    @property
    def delta(self):
        """Grid spacing in the original height units."""
        return self.mapper.delta

    def edges(self):
        """Edges directed from lower to higher height."""
        return self.mapper.edges()

    def has_edge(self, u, v):
        """Return whether the uphill edge from `u` to `v` exists."""
        return self.mapper.has_edge(u, v)

    def incident_edges(self, v):
        """Incoming and outgoing edges at v, each directed uphill."""
        yield from self.mapper.in_edges(v)
        yield from self.mapper.out_edges(v)

    # ---- points ---------------------------------------------------------
    def normalize(self, p):
        """Validate a point and give it a consistent representation.

        An edge endpoint and its vertex refer to the same point. Returning
        the vertex in both cases lets adjacent edges agree on their shared
        endpoint. Interior points keep their edge and exact height.

        Parameters:
            p : tuple
                A vertex point from `V` or an edge point from `E`.

        Returns:
            tuple
                A vertex point if `p` is at a vertex or edge endpoint.
                Otherwise, an edge point with a Fraction height.

        Raises:
            ValueError
                If the tag, vertex, or edge is unknown, or the height lies
                outside the edge.

        Examples:
            >>> F = FunctionalDistortionGraph.from_edges({"u": 2, "v": 3}, [("u", "v")])
            >>> edge = ("u", "v")
            >>> F.normalize(E(edge, 3))
            ('v', 'v')
            >>> F.normalize(E(edge, "5/2"))
            ('e', ('u', 'v'), Fraction(5, 2))
        """
        if p[0] == "v":
            if p[1] not in self.f:
                raise ValueError(f"unknown vertex {p[1]!r}")
            return ("v", p[1])
        if p[0] != "e":
            raise ValueError(f"not a point: {p!r}")
        _, e, h = p
        h = _frac(h)
        if not self.has_edge(*e):
            raise ValueError(f"unknown edge {e!r}")
        lo, hi = e
        if h == self.f[lo]:
            return ("v", lo)
        if h == self.f[hi]:
            return ("v", hi)
        if not (self.f[lo] < h < self.f[hi]):
            raise ValueError(f"height {h} is outside edge {e!r}")
        return ("e", e, h)

    def height(self, p):
        """Return a normalized point's height as a Fraction in grid units.

        For a vertex, look up its height in `f`. For an edge interior,
        use the height stored in the point tuple. Use `normalize` first
        if the point needs validation.
        """
        if p[0] == "v":
            return Fraction(self.f[p[1]])
        return p[2]

    def edges_of(self, p):
        """Edges containing a normalized point (its own edge, or the edges at a vertex)."""
        return [p[1]] if p[0] == "e" else list(self.incident_edges(p[1]))

    def common_edge(self, p, q):
        """The unique edge containing both distinct normalized points, or None."""
        both = set(self.edges_of(p)) & set(self.edges_of(q))
        if not both:
            return None
        if len(both) > 1:
            raise ValueError("common_edge called on equal vertices")
        return next(iter(both))

    def map_to(self, target, vertex_image, routes=None, reparameterizations=None):
        """Build a continuous piecewise-linear map from this graph to `target`.

        Parameters:
            target : FunctionalDistortionGraph
                The graph containing the images of source points.
            vertex_image : dict
                Maps every source vertex to a target point. Use a bare
                vertex label, ``V(v)``, or ``E(edge, h)``. For labels that
                look like tagged point tuples, use ``V(label)``.
            routes : dict, optional
                Maps an uphill source edge to its intermediate target
                points in traversal order. The endpoint images come from
                `vertex_image`. Omit an edge's route to go directly between
                those images. Consecutive points must lie on a common target
                edge or be equal, which gives a constant piece.
            reparameterizations : dict, optional
                Maps a source edge to the parameter for each intermediate
                route point. The parameter runs from 0 at the lower endpoint
                to 1 at the upper endpoint. Values must increase strictly
                inside (0, 1). If omitted, the route pieces take equal
                portions of the source edge.

        Returns:
            dict
                Map data with ``source``, ``target``, ``vertex_image``, and
                ``segments``. Each source edge's segments are ordered tuples
                ``(s0, s1, p0, p1)``. The image moves linearly from target
                point ``p0`` to ``p1`` as the source parameter goes from
                ``s0`` to ``s1``. Pass the dictionary to `evaluate_map`
                or to the distortion computation.

        Raises:
            ValueError
                If a vertex image is missing, a target point is invalid,
                a reparameterization is invalid, or a route jumps between target edges.
            TypeError
                If a height or reparameterization parameter is a float.

        Examples:
            Map one edge onto a two-edge path. The image reaches vertex
            b one-third of the way along the source edge.

            >>> F = FunctionalDistortionGraph.from_edges({"u": 0, "v": 1}, [("u", "v")])
            >>> G = FunctionalDistortionGraph.from_edges(
            ...     {"a": 0, "b": 1, "c": 2}, [("a", "b"), ("b", "c")]
            ... )
            >>> edge = ("u", "v")
            >>> phi = F.map_to(G, {"u": "a", "v": "c"}, routes={edge: ["b"]},
            ...                reparameterizations={edge: ["1/3"]})
            >>> evaluate_map(phi, E(edge, "1/3"))
            ('v', 'b')
        """
        routes = routes or {}
        reparameterizations = reparameterizations or {}

        def point(value):
            if isinstance(value, tuple) and len(value) in (2, 3) and value[0] in ("v", "e"):
                return target.normalize(value)
            return target.normalize(V(value))

        images = {v: point(image) for v, image in vertex_image.items()}
        missing = self.f.keys() - images.keys()
        if missing:
            raise ValueError(f"missing vertex images: {missing!r}")

        segments = {}
        for edge in self.edges():
            u, v = edge
            points = [images[u], *map(point, routes.get(edge, ())), images[v]]
            reparameterization = reparameterizations.get(edge)
            if reparameterization is None:
                times = [Fraction(i, len(points) - 1) for i in range(len(points))]
            else:
                if len(reparameterization) != len(points) - 2:
                    raise ValueError(f"edge {edge!r}: reparameterization and route lengths differ")
                times = [Fraction(0), *map(_frac, reparameterization), Fraction(1)]

            pieces = []
            for s0, s1, p0, p1 in zip(times, times[1:], points, points[1:]):
                if s0 >= s1:
                    raise ValueError(f"edge {edge!r}: reparameterization must increase strictly from 0 to 1")
                if p0 != p1 and target.common_edge(p0, p1) is None:
                    raise ValueError(f"edge {edge!r}: {p0!r} and {p1!r} are not on one target edge")
                pieces.append((s0, s1, p0, p1))
            segments[edge] = pieces
        return dict(source=self, target=target, vertex_image=images, segments=segments)


def evaluate_map(mapping, p):
    """Return the image of a source point under a piecewise-linear map.

    A source vertex uses its recorded image. An interior point uses the
    map segment covering its position along the source edge. For a worked
    example, see `FunctionalDistortionGraph.map_to`.

    Parameters:
        mapping : dict
            Map data returned by `FunctionalDistortionGraph.map_to`.
        p : tuple
            Source point created with `V` or `E`.

    Returns:
        tuple
            A normalized target point. Interior images are interpolated
            exactly using Fraction arithmetic.
    """
    source, target = mapping["source"], mapping["target"]
    p = source.normalize(p)
    if p[0] == "v":
        return mapping["vertex_image"][p[1]]

    _, edge, height = p
    s = height - source.f[edge[0]]
    for s0, s1, p0, p1 in mapping["segments"][edge]:
        if s0 <= s <= s1:
            if p0 == p1:
                return p0
            target_edge = target.common_edge(p0, p1)
            h0, h1 = target.height(p0), target.height(p1)
            return target.normalize(E(target_edge, h0 + (s - s0) / (s1 - s0) * (h1 - h0)))
    raise ValueError("segments do not cover the edge")
