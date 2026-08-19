"""Small igraph compatibility layer for the bundled MetaD2A checkpoint.

The original CDENAG code only needs a directed graph, vertex attributes,
predecessors, and successors.  This module also accepts python-igraph's pickle
constructor so ``nasbench201.pt`` can be loaded when python-igraph is absent.
It is not a general replacement for python-igraph.
"""

from __future__ import absolute_import, division, print_function


class _Vertex(object):
    def __init__(self, attributes=None):
        self.attributes = dict(attributes or {})

    def __getitem__(self, key):
        return self.attributes[key]

    def __setitem__(self, key, value):
        self.attributes[key] = value


class _VertexSequence(object):
    def __init__(self, graph):
        self.graph = graph

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.graph._vertices[index]
        return self.graph._vertices[int(index)]

    def __len__(self):
        return len(self.graph._vertices)


class Graph(object):
    def __init__(
        self,
        n=0,
        edges=None,
        directed=False,
        graph_attrs=None,
        vertex_attrs=None,
        edge_attrs=None,
        **kwargs
    ):
        if isinstance(n, bool) and edges is None:
            directed, n = n, 0
        self.directed = bool(directed or kwargs.get("directed", False))
        self.graph_attrs = dict(graph_attrs or {})
        self._vertices = [_Vertex() for _ in range(int(n or 0))]
        self._edges = []
        self.vs = _VertexSequence(self)
        for source, target in list(edges or []):
            self.add_edge(source, target)
        for name, values in dict(vertex_attrs or {}).items():
            for index, value in enumerate(values):
                if index < len(self._vertices):
                    self._vertices[index][name] = value

    def add_vertices(self, number):
        self._vertices.extend(_Vertex() for _ in range(int(number)))

    def add_edge(self, source, target):
        edge = (int(source), int(target))
        if edge not in self._edges:
            self._edges.append(edge)

    def vcount(self):
        return len(self._vertices)

    def successors(self, vertex):
        vertex = int(vertex)
        return [target for source, target in self._edges if source == vertex]

    def predecessors(self, vertex):
        vertex = int(vertex)
        return [source for source, target in self._edges if target == vertex]

