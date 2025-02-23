# Parts of this file were copy-pasted from graph_generate.py by Colton Griffin and Aneesh Vinod Khilnani (https://github.com/chemfinal-dot/2021-Sinclair-Project)

import numpy as np
from typing import *
from graph_generate import Graph

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

def compress(matrix):
    """
    Obtains the matrix A, defined as A_ij = matrix_{i * n + i, j * n + j}
    """

    width, _ = matrix.size
    n = int(width ** 0.5)
    return np.matrix([[matrix[i * n + i, j * n + j] for j in range(0, n)] for i in range(0, n)])
