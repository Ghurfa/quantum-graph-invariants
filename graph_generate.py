# Parts of this file were copy-pasted from graph_generate.py by Colton Griffin and Aneesh Vinod Khilnani (https://github.com/chemfinal-dot/2021-Sinclair-Project)

import numpy as np
import random
from typing import *

def edges(graph: Dict[int, List[int]]):
    """
    Return the edge list of a graph and the edge list of the complementary graph

    Input: Adjacency list
    Note that if [i,j] is an edge, then [j,i] is also considered an edge.
    """
    edge = []
    edge_complement = []
    for node in graph:
        for neighbour in graph[node]:
            edge.append([node, neighbour])
    for i in range(len(graph)):
        for j in range(len(graph)):
            if [i,j] not in edge and i != j:
                edge_complement.append([i,j])
    return edge, edge_complement

def complete_edges(graph: Dict[int, List[int]]):
    """
    Converts unidirectional edges of a graph to bidirectional

    Input: Adjacency list
    """

    for x in range(len(graph)):
        for i in range(len(graph)):
            if i in graph[x] and x not in graph[i]:
                graph[i].append(x)
    return graph

def cycle(n: int):
    """
    Creates a cycle graph of n nodes
    """

    graph = dict(zip([i for i in range(n)], [[] for i in range(n)]))
    for x in range(n):
        graph[x].append((x+1) % n)
        graph[x].append((x-1) % n)
    return graph

def line(n: int):
    """
    Creates a line graph of n nodes
    """

    if n == 0:
        return dict()
    elif n == 1:
        return {0: []}

    graph = {0: [1], n - 1: {n - 2}}
    for i in range(1, n - 1):
        graph[i] = [i - 1, i + 1]

    return graph

def dense_graph(n):
    """
    Creates a graph where each possible edge is given a 50% chance of being assigned.
    """

    graph = dict(zip([i for i in range(n)], [[] for i in range(n)]))
    for x in range(n):
        for i in range(x+1,n):
            if random.random() < 0.8:
                graph[x].append(i)
    return complete_edges(graph)

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

def adjacency_matrix(graph: Dict[int, List[int]]):
    """
    Creates the adjacency matrix of a graph

    Input: Adjacency list
    """

    n = len(graph)
    edge_set, edge_complement = edges(graph)
    E = np.zeros((n, n))
    E_complement = np.zeros((n, n))
    for x in edge_set:
        E[x[0], x[1]] = 1
    for x in edge_complement:
        E_complement[x[0], x[1]] = 1
    return E, E_complement
