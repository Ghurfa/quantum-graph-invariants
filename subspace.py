from __future__ import annotations
import numpy as np
from typing import *
from graph_generate import Graph

class Subspace:
    def __init__(self, n: int):
        self._n = n
        self.basis = []
        self.constraints = []

    @property
    def n(self) -> int:
        return self._n
    
    def contains(self, mat: np.matrix) -> bool:
        if mat.shape != (self._n, self._n):
            return False
        
        return all(np.vdot(mat, constraint) == 0 for constraint in self.constraints)

    def is_subspace_of(self, other: Subspace) -> bool:
        if self._n != other._n:
            return False
        
        return all(other.contains(bvec) for bvec in self.basis)

def gen_rand(num_constraints: int) -> Subspace:
    '''
    Generate a subspace with k constraints. 
    '''
    raise NotImplementedError()

def mn(n: int) -> Subspace:
    '''
    Generate the complete vector space M_n
    '''

    ret = Subspace(n)

    for i in range(n):
        for j in range(n):
            mat = np.zeros([n, n])
            mat[i, j] = 1
            ret.basis.append(mat)

    return ret


def sg(graph: Graph) -> Subspace:
    '''
    Generate the subspace S_gamma, where not(i ~ j or i = j) implies X_ij = 0
    '''
    n = graph.n
    ret = Subspace(n)
    edges, non_edges = graph.edges()
    
    for i in range(n):
        mat = np.zeros([n, n])
        mat[i, i] = 1
        ret.basis.append(mat)

    for (i, j) in edges:
        mat = np.zeros([n, n])
        mat[i, j] = 1
        ret.basis.append(mat)

    for (i, j) in non_edges:
        mat = np.zeros([n, n])
        mat[i, j] = 1
        ret.constraints.append(mat)
    
    return ret

def eg(graph: Graph) -> Subspace:
    '''
    Generate the subspace E_gamma, which is like S_gamma except the diagonal entries are equal
    '''

    n = graph.n
    ret = Subspace(n)
    edges, non_edges = graph.edges()
    
    ret.basis += np.identity(n)
    for i in range(1, n):
        mat = np.zeros([n, n])
        mat[0, 0] = 1
        mat[i, i] = -1
        ret.constraints.append(mat)

    for (i, j) in edges:
        mat = np.zeros([n, n])
        mat[i, j] = 1
        ret.basis.append(mat)

    for (i, j) in non_edges:
        mat = np.zeros([n, n])
        mat[i, j] = 1
        ret.constraints.append(mat)
    
    return ret
