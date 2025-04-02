# Parts of this file were copy-pasted from graph_generate.py by Colton Griffin and Aneesh Vinod Khilnani (https://github.com/chemfinal-dot/2021-Sinclair-Project)

from __future__ import annotations
from typing import *
import numpy as np
import random as randy

class Graph:
    def __init__(self, adj_list: Dict[int, List[int]]):
        self.adj_list = adj_list
        self._n = len(adj_list)

    @property
    def n(self) -> int:
        return self._n
    
    @property
    def compl(self) -> Graph:
        n = self.n
        return Graph({i: list(set(range(n)) - set(self.adj_list[i])) for i in range(n)})
    
    def __sub__(self, other: Graph) -> Graph:
        if not(other.is_subgraph_of(self)):
            raise ValueError("Not a subgraph; cannot subtract")
        
        n = self.n
        return Graph({i: list(set(self.adj_list[i]) - set(other.adj_list[i])) for i in range(n)})


    @property
    def edges(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Return the edge list of a graph and the edge list of the complementary graph

        Note that if [i,j] is an edge, then [j,i] is also considered an edge.
        """
        
        edge = []
        edge_complement = []
        for node in self.adj_list:
            for neighbour in self.adj_list[node]:
                edge.append((node, neighbour))
        for i in range(self._n):
            for j in range(self._n):
                if (i, j) not in edge and i != j:
                    edge_complement.append((i, j))
        return edge, edge_complement

    @property
    def adjacency_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates the adjacency matrix of the graph
        """

        n = self._n
        edges, nonedges = self.edges
        E = np.zeros((n, n), dtype=int)
        E_complement = np.zeros((n, n), dtype=int)
        for x in edges:
            E[x[0], x[1]] = 1
        for x in nonedges:
            E_complement[x[0], x[1]] = 1
        return E, E_complement

    @property
    def laplacian_matrix(self) -> np.ndarray:
        """
        Creates the Laplacian matrix of the graph, having -1 at (i, j) if i ~ j and the
        vertex degrees on the diagonal
        """
        n = self._n
        edges, _ = self.edges
        L = np.zeros((n, n), dtype=int)

        for (i, j) in edges:
            L[i, j] = -1
        
        for i in range(n):
            L[i, i] = len(self.adj_list[i])
        
        return L
    
    def is_subgraph_of(self, other: Graph) -> bool:
        if self._n != other._n:
            return False
        
        # Check for each vertex, the neighbor list is a subset of the corresponding neighbor list in other
        for i in range(self._n):
            if not(all(neighbor in other.adj_list[i] for neighbor in self.adj_list[i])):
                return False
        
        return True
    
    def __str__(self):
        return "\n".join(str(i) + ":\t" + " ".join(str(j) for j in self.adj_list[i]) for i in range(self.n))
    
    def __repr__(self):
        return self.__str__()

def cycle(n: int) -> Graph:
    """
    Creates a cycle graph of n nodes
    """

    if n < 3:
        return complete(n)

    adj_list = { i: [(i + 1) % n, (i - 1) % n] for i in range(n) }
    return Graph(adj_list)

def line(n: int) -> Graph:
    """
    Creates a line graph of n nodes
    """

    if n < 3:
        return complete(n)

    adj_list = { 0: [1], n - 1: {n - 2} }
    for i in range(1, n - 1):
        adj_list[i] = [i - 1, i + 1]

    return Graph(adj_list)

def complete(n: int) -> Graph:
    """
    Creates a complete graph of n nodes
    """

    adj_list = { i: [j for j in range(n) if j != i] for i in range(n) }
    return Graph(adj_list)

def independent(n: int) -> Graph:
    """
    Creates a graph of n nodes with no edges
    """

    adj_list = { i: [] for i in range(n)}
    return Graph(adj_list)

def from_edge_list(n: int, *edge_list: List[Tuple[int, int]]) -> Graph:
    """
    Creates a graph with the given number of nodes and edge list

    Edges need only be specified in one direction
    """
    adj_list = { i: [] for i in range(n) }
    for (i, j) in edge_list:
        adj_list[i].append(j)
        adj_list[j].append(i)
    return Graph(adj_list)

def random(n: int, density: float) -> Graph:
    """
    Generate a random graph with the given density
    """

    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if randy.random() < density:
                edge_list.append((i, j))
    
    return from_edge_list(n, *edge_list)
