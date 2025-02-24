# Parts of this file were copy-pasted from graph_generate.py by Colton Griffin and Aneesh Vinod Khilnani (https://github.com/chemfinal-dot/2021-Sinclair-Project)

import numpy as np
from typing import *
from graph_generate import Graph

def e_matrix(n: int, i: int, j: int):
    """
    Creates a matrix that is all zeroes except for a one at (i, j)
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

class SimpleMatrix:
    def __init__(self, mat: np.matrix, precision: int = 2):
        self.data = mat
        self.precision = precision
    
    def _entry_str(self, i, j):
        col_width = 3 + self.precision
        val = round(self.data[i, j], self.precision)
        ret = f"{val:.{self.precision}f}".rjust(col_width)
        if val == 0:
            return "  ." + " " * (col_width - 3)
        if ret[-self.precision:] == "0" * self.precision:
            ret = ret[:-self.precision] + " " * self.precision
        return ret
    
    def __str__(self):
        width, height = self.data.shape
        def row(i):
            return " ".join(self._entry_str(i, j) for j in range(0, width))
        return "\n".join(row(i) for i in range(0, height))
    
    def __repr__(self):
        return self.__str__()


class SimpleChoiMatrix(SimpleMatrix):
    def __init__(self, mat: np.matrix, precision: int = 2):
        SimpleMatrix.__init__(self, mat, precision)
        self.n = int(mat.shape[0] ** 0.5)
    
    def __str__(self):
        col_width = 3 + self.precision
        def col_sect(i, j):
            return " ".join(self._entry_str(i, j * self.n + k) for k in range(0, self.n))
        def row(i):
            return " | ".join(col_sect(i, j) for j in range(0, self.n))
        def row_sect(i):
            return "\n".join(row(i * self.n + j) for j in range(0, self.n))
        row_sects = [row_sect(i) for i in range(0, self.n)]
        return ("\n" + "-" * ((col_width + 1) * (self.n * self.n) + (3 - 1) * self.n - 3) + "\n").join(row_sects)

    
def compress(matrix: SimpleChoiMatrix):
    """
    Obtains the matrix A, defined as A_ij = matrix_{i * n + i, j * n + j}
    """

    if not(matrix is SimpleChoiMatrix):
        raise ValueError("compress() expects a Choi (n^2 by n^2) matrix")
    
    n = matrix.n
    return SimpleMatrix(np.matrix([[matrix[i * n + i, j * n + j] for j in range(0, n)] for i in range(0, n)]))
