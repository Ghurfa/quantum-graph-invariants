# Parts of this file were copy-pasted from graph_generate.py by Colton Griffin and Aneesh Vinod Khilnani (https://github.com/chemfinal-dot/2021-Sinclair-Project)

import numpy as np
from typing import *

class Graph:
    def __init__(self, adj_list: Dict[int, List[int]]):
        self._adj_list = adj_list
        self._n = len(adj_list)

    @classmethod
    def from_edge_list(cls, n, *edge_list: List[Tuple[int, int]]):
        adj_list = { i: [] for i in range(0, n) }
        for (i, j) in edge_list:
            adj_list[i].append(j)
            adj_list[j].append(i)
        return cls(adj_list)

    def n(self) -> int:
        return self._n

    def edges(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Return the edge list of a graph and the edge list of the complementary graph

        Input: Adjacency list
        Note that if [i,j] is an edge, then [j,i] is also considered an edge.
        """
        edge = []
        edge_complement = []
        for node in self._adj_list:
            for neighbour in self._adj_list[node]:
                edge.append([node, neighbour])
        for i in range(self._n):
            for j in range(self._n):
                if [i,j] not in edge and i != j:
                    edge_complement.append([i,j])
        return edge, edge_complement

def cycle(n: int) -> Graph:
    """
    Creates a cycle graph of n nodes
    """

    adj_list = dict(zip([i for i in range(n)], [[] for i in range(n)]))
    for x in range(n):
        adj_list[x].append((x+1) % n)
        adj_list[x].append((x-1) % n)
    return Graph(adj_list)

def line(n: int) -> Graph:
    """
    Creates a line graph of n nodes
    """

    if n == 0:
        return dict()
    elif n == 1:
        return {0: []}

    adj_list = { 0: [1], n - 1: {n - 2} }
    for i in range(1, n - 1):
        adj_list[i] = [i - 1, i + 1]

    return Graph(adj_list)

def complete(n: int) -> Graph:
    """
    Creates a complete graph of n nodes
    """

    adj_list = { i: [j for j in range(0, n) if j != i] for i in range(0, n) }
    return Graph(adj_list)

def independent(n: int) -> Graph:
    """
    Creates a graph of n nodes with no edges
    """

    adj_list = { i: [] for i in range(0, n)}
    return Graph(adj_list)

def e_matrix(n: int, i: int, j: int):
    """
    Creates matrix that is all zeroes except for a one at (i, j)
    """

    E = np.zeros((n,n))
    E[i, j] = 1
    return E

def delta_matrix(n: int):
    """
    Creates a matrix that is the sum of (e (x) e) where e is each of the e-matrices of size n
    """

    Delta = np.zeros((n**2,n**2))
    for i in range(n):
        for j in range(n):
            E = e_matrix([i,j], n)
            Delta = np.add(Delta, np.kron(E,E))
    return Delta

def adjacency_matrix(graph: Graph):
    """
    Creates the adjacency matrix of a graph

    Input: Adjacency list
    """

    n = graph._n
    edges, nonedges = graph.edges
    E = np.zeros((n, n))
    E_complement = np.zeros((n, n))
    for x in edges:
        E[x[0], x[1]] = 1
    for x in nonedges:
        E_complement[x[0], x[1]] = 1
    return E, E_complement
