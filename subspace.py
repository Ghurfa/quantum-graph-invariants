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
    
    def contains(self, mat: np.array) -> bool:
        if mat.shape != (self._n, self._n):
            return False
        
        return all(np.trace(mat @ constraint.T) == 0 for constraint in self.constraints)

    def is_subspace_of(self, other: Subspace) -> bool:
        if self._n != other._n:
            return False
        
        return all(other.contains(bvec) for bvec in self.basis)
    
    @property
    def perp(self) -> Subspace:
        ret = Subspace(self.n)
        ret.constraints = self.basis
        ret.basis = self.constraints
        return ret
    
    def ensure_valid(self):
        """
        Checks whether the subspace is a valid matricial system, i.e. that self.basis and self.constraints
        are linearly independent sets, that the constraints form a basis for S^perp, and that it is closed
        under taking the adjoint
        """

        n = self.n

        if not(self.contains(np.identity(n))):
            raise ValueError("Does not contain identity")

        if len(self.basis) + len(self.constraints) != n * n:
            raise ValueError("Basis or constraints are incomplete (too few)")
        
        super_space = Subspace(n)
        super_space.basis = self.basis + self.constraints

        for i in range(n):
            for j in range(n):
                mat = np.zeros([n, n])
                mat[i, j] = 1
                if not(super_space.contains(mat)):
                    raise ValueError("Basis or constraints are incomplete (S + S^perp != M_n)")
        
        for bvec in self.basis:
            for bvec_other in self.basis:
                if np.array_equal(bvec.conj().T, bvec_other):
                    break
            else:
                raise ValueError("Basis missing adjoint of basis vector (S is probably not closed under adjoint)")
        

def gen_rand(num_constraints: int) -> Subspace:
    """
    Generate a subspace with k constraints. 
    """

    raise NotImplementedError()

def mn(n: int) -> Subspace:
    """
    Generate the complete vector space M_n
    """

    ret = Subspace(n)

    for i in range(n):
        for j in range(n):
            mat = np.zeros([n, n])
            mat[i, j] = 1
            ret.basis.append(mat)

    return ret


def sg(graph: Graph) -> Subspace:
    """
    Generate the subspace S_gamma, where not(i ~ j or i = j) implies X_ij = 0
    """

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
    """
    Generate the subspace E_gamma, which is like S_gamma except the diagonal entries are equal
    """

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
