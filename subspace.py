from __future__ import annotations
from functools import reduce
import numpy as np
from typing import *

from graph_generate import Graph
import matrix_manip as mm


class Subspace:
    def __init__(self, n: int):
        self._n = n
        self.basis = []
        self.constraints = []

    @property
    def n(self) -> int:
        return self._n
    
    @property
    def perp(self) -> Subspace:
        ret = Subspace(self.n)
        ret.constraints = self.basis
        ret.basis = self.constraints
        return ret
    
    def contains(self, mat: np.array) -> bool:
        return not(is_independent(mat, self.basis))

    def is_subspace_of(self, other: Subspace) -> bool:
        if self._n != other._n:
            return False
        
        return all(other.contains(bvec) for bvec in self.basis)
    
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
            raise ValueError("Basis or constraints are incomplete (wrong number)")
        
        super_space = Subspace(n)
        super_space.basis = self.basis + self.constraints

        for i in range(n):
            for j in range(n):
                mat = np.zeros([n, n])
                mat[i, j] = 1
                if not(super_space.contains(mat)):
                    raise ValueError("Basis or constraints are incomplete (S + S^perp != M_n)")
        
        for bvec in self.basis:
            adjoint = bvec.conj().T
            for bvec_other in self.basis:
                if np.array_equal(adjoint, bvec_other) or np.array_equal(-adjoint, bvec_other):
                    break
            else:
                raise ValueError("Basis missing adjoint of basis vector (S is probably not closed under adjoint)")

def mn(n: int) -> Subspace:
    """
    Generate the complete vector space M_n with the standard basis
    """

    ret = Subspace(n)

    for i in range(n):
        for j in range(n):
            ret.basis.append(mm.e_matrix(n, i, j))

    return ret


def sg(graph: Graph) -> Subspace:
    """
    Generate the subspace S_gamma, where not(i ~ j or i = j) implies X_ij = 0
    """

    n = graph.n
    ret = Subspace(n)
    edges, non_edges = graph.edges()
    
    for i in range(n):
        ret.basis.append(mm.e_matrix(n, i, i))

    for (i, j) in edges:
        ret.basis.append(mm.e_matrix(n, i, j))

    for (i, j) in non_edges:
        ret.constraints.append(mm.e_matrix(n, i, j))
    
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
        ret.basis.append(mm.e_matrix(n, i, j))

    for (i, j) in non_edges:
        ret.constraints.append(mm.e_matrix(n, i, j))
    
    return ret

def is_independent(matrix: np.ndarray, basis: List[np.ndarray]) -> bool:
    """
    Checks if matrix is linearly independent from the given basis
    """

    flattened_basis = [mat.flatten() for mat in basis]
    flattened_new = matrix.flatten()
    
    matrix_stack = np.vstack(flattened_basis + [flattened_new])
    rank_before = np.linalg.matrix_rank(np.vstack(flattened_basis))
    rank_after = np.linalg.matrix_rank(matrix_stack)
    
    return rank_after > rank_before

def is_h_ortho(matrix: np.ndarray, basis: List[np.ndarray]) -> bool:
    """
    Checks if the adjoint of matrix is orthogonal to the given basis and that matrix != 0
    """

    return matrix.any() and all((np.trace(bvec @ matrix) == 0) for bvec in basis)

def orthogonalize(matrix: np.ndarray, basis: List[np.ndarray]) -> np.ndarray:
    """
    Modifies (adjoint of) matrix to be orthogonal to the given basis while ensuring integer entries
    """

    for bvec in basis:
        num = np.trace(bvec @ matrix)
        if num == 0:
            continue

        denom = np.trace(bvec @ bvec.conj().T)
        gcd = np.gcd(num, denom)
        num //= gcd
        denom //= gcd

        matrix = matrix * denom - num * bvec

        mat_gcd = np.gcd.reduce(matrix, axis=(0, 1))
        matrix //= mat_gcd

    if matrix[0, 0] == 130:
        x = 2
        pass
    mat_gcd = np.gcd.reduce(matrix, axis=(0, 1))
    return matrix // mat_gcd

def random_basis(n: int, low=-10, high=10, density=0.3) -> List[np.ndarray]:
    """
    Generates a randomized basis for M_n
    """

    basis = [np.identity(n).astype(int)]
    num_nonzero = max(1, int(density * n * n))
    
    while len(basis) < n * n:
        matrix = np.zeros((n, n), dtype=int)
        indices = np.random.choice(n * n, num_nonzero, replace=False)
        for index in indices:
            i, j = divmod(index, n)
            matrix[i, j] = np.random.randint(low, high + 1)

        if not(is_independent(matrix, basis)):   # Regenerate if matrix is in span of basis already
            continue

        matrix = orthogonalize(matrix, basis)
        
        adjoint = matrix.conj().T
        if np.array_equal(matrix, adjoint) or np.array_equal(matrix, -adjoint):
            basis.append(matrix)
        elif is_h_ortho(adjoint, basis + [matrix]):
            basis.append(matrix)
            basis.append(adjoint)
        else:
            matrix = matrix + adjoint
            if is_h_ortho(matrix, basis):
                basis.append(matrix)
    
    return basis

def random_s1_s2(n: int, low=-10, high=10, density=0.3) -> Tuple[Subspace, Subspace, Subspace]:
    """
    Generates random subspaces S1 and S2 such that I_n \in S_2 \subseteq S1 \subseteq M_n.

    Returns S1, S2, and S1^perp + S2
    """

    # Start with a randomized basis for M_n. Assume basis[0] = I_n
    starting_basis = random_basis(n, low, high, density)

    # For each basis vector pair (A, A*), remove A* (will be added back later)
    bvecs = [starting_basis.pop(0)]
    while len(starting_basis) > 0:
        bvec = starting_basis.pop()
        if (len(starting_basis) > 0) and np.array_equal(bvec.conj().T, starting_basis[-1]):
            starting_basis.pop()
        bvecs.append(bvec)
    
    np.random.shuffle(bvecs[1:])

    # Partition basis vectors. S2 will be formed from matrices 0 to a (exclusive)
    # +1 and -1 are temporary because cvxopt is being weird when s1 = s2 (and weird in general, ig)
    a = np.random.randint(1, len(bvecs) - 1)
    b = np.random.randint(a + 1, len(bvecs))

    def extract_basis(start, stop) -> List[np.ndarray]:
        ret = []
        for i in range(start, stop):
            bvec = bvecs[i]
            if np.array_equal(bvec, bvec.conj().T) or np.array_equal(bvec, -bvec.conj().T):
                ret.append(bvec)
            else:
                ret.append(bvec)
                ret.append(bvec.conj().T)
        return ret

    s2 = Subspace(n)
    s2.basis = extract_basis(0, a)
    s2.constraints = extract_basis(a, len(bvecs))

    s1 = Subspace(n)
    s1.basis = extract_basis(0, b)
    s1.constraints = extract_basis(b, len(bvecs))

    s1pps2 = Subspace(n)
    s1pps2.basis = s2.basis + s1.constraints
    s1pps2.constraints = extract_basis(a, b)

    return s1, s2, s1pps2
