from __future__ import annotations
import numpy as np
from typing import *

import graph_generate as gg
from graph_generate import Graph
import matrix_manip as mm

class Subspace:
    def __init__(self, n: int):
        self.n = n
        self.s0 = []
        self.constraints = []
    
    @property
    def basis(self) -> Iterable[np.ndarray]:
        yield np.identity(self.n)
        for bvec in self.s0:
            yield bvec
    
    @property
    def compl(self) -> Subspace:
        ret = Subspace(self.n)
        ret.s0 = self.constraints
        ret.constraints = self.s0
        return ret
    
    def contains(self, mat: np.ndarray) -> bool:
        return all(np.isclose(np.trace(constraint @ mat.T), 0) for constraint in self.constraints)

    def is_subspace_of(self, other: Subspace) -> bool:
        if self.n != other.n:
            return False
        
        return all(other.contains(bvec) for bvec in self.s0)
    
    def ensure_valid(self):
        """
        Checks whether the subspace is a valid matricial system, i.e. that self.basis and self.constraints
        are linearly independent sets, that the constraints form a basis for S^perp, and that it constains
        the identity & is closed under taking the adjoint
        """

        n = self.n

        assert all(np.isclose(np.trace(bvec), 0) for bvec in self.s0), "A S0 vector is not traceless"
        assert all(np.isclose(np.trace(bvec), 0) for bvec in self.constraints), "A constraint vector is not traceless"

        assert len(self.s0) + len(self.constraints) == n * n - 1, "Basis or constraints are incomplete (wrong number)"
        
        assert is_independent([np.identity(n)] + self.s0 + self.constraints), "Basis or constraints are incomplete (S + S^perp != M_n)"
        
        for bvec in self.s0:
            adjoint = bvec.conj().T
            assert any(np.allclose(adjoint, bvec_other) or np.allclose(-adjoint, bvec_other) for bvec_other in self.s0), \
                "Basis missing adjoint of basis vector (cannot verify closure under adjoint)"
            
    def __str__(self):
        integral = ('int' in str(self.s0[0].dtype)) if len(self.s0) > 0 else ('int' in str(self.constraints[0].dtype))
        precision = 0 if integral else 2
        ret = "BASIS:\n" + "\n----------------\n".join(str(mm.SimpleMatrix(bvec, precision)) for bvec in self.basis) + \
              "\n\nCONSTRAINTS:\n" + "\n\n".join(str(mm.SimpleMatrix(const, precision)) for const in self.constraints)
        return ret
    
    def __repr__(self):
        return self.__str__()

def tensor(s1: Subspace, s2: Subspace) -> Subspace:
    """
    Generate the tensor product subspace of s1 and s2
    """

    ret = Subspace(s1.n * s2.n)
    ret.s0 = [np.kron(b1, b2) for b1 in s1.s0 for b2 in s2.s0] + \
             [np.kron(np.identity(s1.n), b2) for b2 in s2.s0] + \
             [np.kron(b1, np.identity(s2.n)) for b1 in s1.s0]
    ret.constraints =   [np.kron(c1, c2) for c1 in s1.constraints for c2 in s2.constraints] + \
                        [np.kron(b1, c2) for b1 in s1.basis for c2 in s2.constraints] + \
                        [np.kron(c1, b2) for c1 in s1.constraints for b2 in s2.basis]

    return ret

def tensor2(s1: Subspace, s2: Subspace) -> Subspace:
    """
    Generate the tensor product subspace of s1 and s2, but with the identity in the basis
    """

    ret = Subspace(s1.n * s2.n)
    ret.s0 =    [np.kron(b1, b2) for b1 in s1.s0 for b2 in s2.s0] + \
                [np.kron(b1, c2) for b1 in s1.basis for c2 in s2.constraints] + \
                [np.kron(c1, b2) for c1 in s1.constraints for b2 in s2.basis] + \
                [np.kron(np.identity(s1.n), b2) for b2 in s2.s0] + \
                [np.kron(b1, np.identity(s2.n)) for b1 in s1.s0]
    ret.constraints =   [np.kron(c1, c2) for c1 in s1.constraints for c2 in s2.constraints]

    return ret

def direct_union(s1: Subspace, s2: Subspace) -> Subspace:
    """
    Generate the direct sum subspace of s1 and s2
    """
    
    n = s1.n + s2.n
    ret = Subspace(n)
    ret.s0 = [np.pad(b1, ((0, s2.n), (0, s2.n)), 'constant', constant_values=(0, 0)) for b1 in s1.s0] + \
             [np.pad(b2, ((s1.n, 0), (s1.n, 0)), 'constant', constant_values=(0, 0)) for b2 in s2.s0] + \
             [np.pad(np.identity(s1.n), ((0, s2.n), (0, s2.n)), 'constant', constant_values=(0, 0)) / s1.n -
              np.pad(np.identity(s2.n), ((s1.n, 0), (s1.n, 0)), 'constant', constant_values=(0, 0)) / s2.n]
    ret.constraints =   [np.pad(c1, ((0, s2.n), (0, s2.n)), 'constant', constant_values=(0, 0)) for c1 in s1.constraints] + \
                        [np.pad(c2, ((s1.n, 0), (s1.n, 0)), 'constant', constant_values=(0, 0)) for c2 in s2.constraints] + \
                        [mm.e_matrix(n, i, j) for i in range(s1.n, n) for j in range(0, s1.n)] + \
                        [mm.e_matrix(n, i, j) for j in range(s1.n, n) for i in range(0, s1.n)]

    return ret

def complete_union(s1: Subspace, s2: Subspace) -> Subspace:
    """
    Generate the complete sum subspace of s1 and s2
    """

    n = s1.n + s2.n
    ret = Subspace(n)
    ret.s0 =    [np.pad(b1, ((0, s2.n), (0, s2.n)), 'constant', constant_values=(0, 0)) for b1 in s1.s0] + \
                [np.pad(b2, ((s1.n, 0), (s1.n, 0)), 'constant', constant_values=(0, 0)) for b2 in s2.s0] + \
                [np.pad(np.identity(s1.n), ((0, s2.n), (0, s2.n)), 'constant', constant_values=(0, 0)) / s1.n -
                 np.pad(np.identity(s2.n), ((s1.n, 0), (s1.n, 0)), 'constant', constant_values=(0, 0)) / s2.n] + \
                [mm.e_matrix(n, i, j) for i in range(s1.n, n) for j in range(0, s1.n)] + \
                [mm.e_matrix(n, i, j) for j in range(s1.n, n) for i in range(0, s1.n)]
    ret.constraints =   [np.pad(c1, ((0, s2.n), (0, s2.n)), 'constant', constant_values=(0, 0)) for c1 in s1.constraints] + \
                        [np.pad(c2, ((s1.n, 0), (s1.n, 0)), 'constant', constant_values=(0, 0)) for c2 in s2.constraints]
    return ret
    # return direct_union(s1.compl, s2.compl).compl

def complete_product(s1: Subspace, s2: Subspace) -> Subspace:
    """
    Generate the complete product subspace of s1 and s2
    """
    
    return tensor(s1.compl, s2.compl).compl

def mn(n: int) -> Subspace:
    """
    Generate the complete vector space M_n with the standard basis
    """

    return from_basis([mm.e_matrix(n, i, j) for i in range(n) for j in range(n)])

def ci(n: int) -> Subspace:
    """
    Generate the subspace CI (complex multiples of the identity)
    """
    return from_basis([np.identity(n)])

def sg(graph: Graph) -> Subspace:
    """
    Generate the subspace S_gamma, where not(i ~ j or i = j) implies X_ij = 0
    """

    n = graph.n
    ret = Subspace(n)
    edges, non_edges = graph.edges
    
    return from_basis([mm.e_matrix(n, i, j) for i in range(n) for j in range(n) if ((i, j) in edges) or (i == j)])

def eg(graph: Graph) -> Subspace:
    """
    Generate the subspace E_gamma, which is like S_gamma except the diagonal entries are equal
    """

    n = graph.n
    edges, _ = graph.edges
    return from_basis([np.identity(n)] + [mm.e_matrix(n, i, j) for (i, j) in edges])

def antilaplacian(graph: Graph) -> Subspace:
    """
    Creates the subspace whose orthogonal space is spanned by the matrix L - tr(L)/n * I_n
    """
    
    n = graph.n

    if all(len(graph.adj_list[i]) == 0 for i in range(n)):
        return mn(n)

    L = graph.laplacian_matrix
    mat = L * n - np.identity(n) * np.trace(L)
    mat_gcd = np.gcd.reduce(mat.astype(int), axis=(0, 1))
    return from_constraints([mat.astype(float) // mat_gcd])

def diag_restricted(sub: Subspace) -> Subspace:
    """
    Given a subspace, creates the same subspace with the additional restriction that the
    diagonal entries are equal
    """

    constraints = sub.constraints.copy()
    n = sub.n
    
    def ortho(i):
        ret = np.zeros((n, n))
        ret[0, 0] = 1
        ret[i, i] = -1
        return ret
    
    extend_basis(constraints, iter(ortho(i) for i in range(1, n)))
    return from_constraints(constraints)

def tk(K: np.ndarray) -> Subspace:
    """
    Given a traceless Hermitian nonzero matrix K, creates the subspace orthogonal to it
    """

    assert not(np.array_equal(K, np.zeros_like(K)))
    return from_constraints([K])

def tks(K: np.ndarray) -> Subspace:
    """
    Given a traceless Hermitian nonzero matrix K, creates the equal-diagonal subspace orthogonal to it
    """

    return diag_restricted(tk(K))

def from_basis(basis: List[np.ndarray]):
    """
    Creates a subspace with the given basis
    """

    validate_partial_basis(basis, True)

    n = basis[0].shape[0]
    ret = Subspace(n)

    new_basis = [np.identity(n)]
    extend_basis(new_basis, iter(basis))
    basis_dim = len(basis)
    extend_basis(new_basis)

    ret.s0 = new_basis[1 : basis_dim]
    ret.constraints = new_basis[basis_dim :]

    ret.ensure_valid()
    return ret

def from_constraints(constraints: List[np.ndarray]):
    """
    Creates a subspace with the given constraints
    """
    
    validate_partial_basis(constraints, False)

    n = constraints[0].shape[0]
    ret = Subspace(n)

    new_basis = [np.identity(n)]
    extend_basis(new_basis, iter(constraints))
    constraints_dim = len(constraints)
    extend_basis(new_basis)

    ret.constraints = new_basis[1 : constraints_dim]
    ret.s0 = new_basis[constraints_dim :]

    ret.ensure_valid()
    return ret

def validate_partial_basis(basis: List[np.ndarray], incl_id: bool):
    """
    Checks that the given basis consists of matrices that are of the same size & are
    nonzero & linearly independent. Also either checks that identity is in their span
    (if incl_id = True) or in their orthogonal space (if incl_id = False)
    """

    assert isinstance(basis, List), "Basis is wrong datatype"
    
    if len(basis) == 0:
        assert not(incl_id)
        return

    shape = basis[0].shape
    assert len(shape) == 2
    n = shape[0]

    assert is_independent(basis), "Basis is not independent"
    assert all(bvec.shape == (n, n) for bvec in basis), "Basis contains nonsquare matrices"
    assert all(bvec.any() for bvec in basis), "Basis contains a zero matrix"
    assert all(any(np.allclose(other, bvec.conj().T) or np.allclose(-other, bvec.conj().T) for other in basis) for bvec in basis), "Basis contains duplicate elements"
    if incl_id:
        assert not(is_independent_from(np.identity(n), basis)), "Basis does not contain the identity"
    else:
        assert is_ortho(np.identity(n), basis), "Basis is not orthogonal to the identity when it should be"

def is_independent(basis: List[np.ndarray]) -> bool:
    flattened = [mat.flatten() for mat in basis]
    rank = np.linalg.matrix_rank(np.vstack(flattened))
    return rank == len(basis)

def is_independent_from(matrix: np.ndarray, basis: List[np.ndarray]) -> bool:
    """
    Checks if matrix is linearly independent from the given basis
    """

    if len(basis) == 0:
        return True

    flattened_basis = [mat.flatten() for mat in basis]
    flattened_new = matrix.flatten()
    
    rank_before = np.linalg.matrix_rank(np.vstack(flattened_basis))

    matrix_stack = np.vstack(flattened_basis + [flattened_new])
    rank_after = np.linalg.matrix_rank(matrix_stack)
    
    return rank_after > rank_before

def is_ortho(matrix: np.ndarray, basis: List[np.ndarray]) -> bool:
    """
    Checks if the matrix is orthogonal to the given basis and that matrix != 0
    """

    return not(np.allclose(matrix, np.zeros_like(matrix))) and all(np.isclose(np.trace(bvec @ matrix.conj().T), 0) for bvec in basis)

def orthogonalize(matrix: np.ndarray, basis: List[np.ndarray]) -> np.ndarray:
    """
    Modifies the matrix to be orthogonal to the given basis via Gram-Schmidt
    """

    matrix = matrix.copy()
    for bvec in basis:
        num = np.trace(bvec @ matrix.T)
        denom = np.trace(bvec @ bvec.conj().T)
        matrix -= bvec * (num / denom)

    return matrix

def extend_basis(ortho_basis: List[np.ndarray], mat_src: Iterator[np.ndarray] = None, mat_standardizer: Callable[[np.ndarray], np.ndarray] = mm.normalize):
    """
    Starting with the given orthogonal matrix basis, extends it in-place by taking matrices from mat_src
    and orthogonalizing them. Ends when mat_src runs out or when the basis is a complete basis for M_n.
    If mat_src is unspecified, then we use the standard basis for M_n
    """

    assert mat_src or (ortho_basis and (len(ortho_basis) > 0))

    if len(ortho_basis) == 0:
        ortho_basis.append(next(mat_src))

    n = ortho_basis[0].shape[0]

    if not(mat_src):
        mat_src = iter(mm.e_matrix(n, i, j) for i in range(n) for j in range(n))
    
    while len(ortho_basis) < n * n:
        matrix = next(mat_src, None)
        if matrix is None:                      # mat_src is exhausted
            return

        matrix = orthogonalize(matrix, ortho_basis)
        if np.allclose(matrix, np.zeros_like(matrix)):  # Regenerate if matrix is in span of basis already
            continue
        if mat_standardizer:
            matrix = mat_standardizer(matrix)
        
        adjoint = matrix.conj().T
        if np.allclose(matrix, adjoint) or np.allclose(matrix, -adjoint):
            ortho_basis.append(matrix)
        elif is_ortho(adjoint, ortho_basis + [matrix]):
            ortho_basis.append(matrix)
            ortho_basis.append(adjoint)
        else:
            matrix = matrix + adjoint
            if mat_standardizer:
                matrix = mat_standardizer(matrix)
            if is_ortho(matrix, ortho_basis):
                ortho_basis.append(matrix)

def random_basis(n: int, density=0.3) -> List[np.ndarray]:
    """
    Genekrates a randomized basis for M_n. Basis[0] = I_n
    """

    basis = [np.identity(n)]
    num_nonzero = max(1, int(density * n * n))
    
    def rand_mat_src() -> Iterator[np.ndarray]:
        while True:
            matrix = np.zeros((n, n))
            indices = np.random.choice(n * n, num_nonzero, replace=False)
            for index in indices:
                i, j = divmod(index, n)
                matrix[i, j] = np.random.random() * 4 - 2
            
            yield matrix

    extend_basis(basis, rand_mat_src(), mm.normalize)
    return basis

def random(n: int) -> Subspace:
    """
    Generates a random subspace
    """

    # Start with a randomized basis for M_n. Assume basis[0] = I_n
    starting_basis = random_basis(n, 0.3)

    # For each basis vector pair (A, A*), remove A* (will be added back later)
    bvecs = [starting_basis.pop(0)]
    while len(starting_basis) > 0:
        bvec = starting_basis.pop()
        if (len(starting_basis) > 0) and np.allclose(bvec.conj().T, starting_basis[-1]):
            starting_basis.pop()
        bvecs.append(bvec)
    
    np.random.shuffle(bvecs[1:])

    # Partition basis vectors. S2 will be formed from matrices 0 (inclusive) to a (exclusive)
    a = np.random.randint(1, len(bvecs))

    def extract_basis(start, stop) -> List[np.ndarray]:
        ret = []
        for i in range(start, stop):
            bvec = bvecs[i]
            if np.allclose(bvec, bvec.conj().T) or np.allclose(bvec, -bvec.conj().T):
                ret.append(bvec)
            else:
                ret.append(bvec)
                ret.append(bvec.conj().T)
        return ret

    sub = Subspace(n)
    sub.s0 = extract_basis(1, a)
    sub.constraints = extract_basis(a, len(bvecs))

    return sub

def get_conjugate_basis(myUni: np.ndarray, basis: List[np.ndarray]) -> List[np.ndarray]:
    """
    Given compatible (semi-)unitary matrix (U* U = I) and a basis { b_i }, return { U* b_i U }
    """

    myUniH = myUni.conj().T
    return [myUniH @ bvec @ myUni for bvec in basis]

def sg1_rotate_sg2(unitary: np.ndarray, sg1: Subspace, sg2: Subspace):
    """
    Given two graph systems and a compatible unitary matrix, define the quantum graph given by Usg1U* + Sg2
    #Jordan L checked this function implementation to his original test implementation.
    """
    myQuantumBasis = get_conjugate_basis(unitary, sg1.basis)
    extend_basis(myQuantumBasis, iter(sg2.basis))
    myQuantumGraph = from_basis(myQuantumBasis)
    return myQuantumGraph

def random2(n: int) -> Subspace:
    graph1 = gg.random(n, np.random.random() ** 2)
    graph2 = gg.random(n, np.random.random() ** 2)
    unitary = mm.rand_uni(n)
    return sg1_rotate_sg2(unitary, sg(graph1), sg(graph2))

def random_s1_s2(n: int, density=0.3) -> Tuple[Subspace, Subspace, Subspace]:
    """
    Generates random subspaces S1 and S2 such that I_n in S_2 subseteq S1 subseteq M_n.

    Returns S1, S2, and S1^perp + S2
    """

    # Start with a randomized basis for M_n. Assume basis[0] = I_n
    starting_basis = random_basis(n, density)

    # For each basis vector pair (A, A*), remove A* (will be added back later)
    bvecs = [starting_basis.pop(0)]
    while len(starting_basis) > 0:
        bvec = starting_basis.pop()
        if (len(starting_basis) > 0) and np.allclose(bvec.conj().T, starting_basis[-1]):
            starting_basis.pop()
        bvecs.append(bvec)
    
    np.random.shuffle(bvecs[1:])

    # Partition basis vectors. S2 will be formed from matrices 0 (inclusive) to a (exclusive)
    a = np.random.randint(2, len(bvecs))
    b = np.random.randint(a, len(bvecs))

    def extract_basis(start, stop) -> List[np.ndarray]:
        ret = []
        for i in range(start, stop):
            bvec = bvecs[i]
            if np.allclose(bvec, bvec.conj().T) or np.allclose(bvec, -bvec.conj().T):
                ret.append(bvec)
            else:
                ret.append(bvec)
                ret.append(bvec.conj().T)
        return ret

    s2 = Subspace(n)
    s2.s0 = extract_basis(1, a)
    s2.constraints = extract_basis(a, len(bvecs))

    s1 = Subspace(n)
    s1.s0 = extract_basis(1, b)
    s1.constraints = extract_basis(b, len(bvecs))

    s1pps2 = Subspace(n)
    s1pps2.s0 = s2.s0 + s1.constraints
    s1pps2.constraints = extract_basis(a, b)

    return s1, s2, s1pps2

def random_tk(n: int) -> Subspace:
    matrix = np.zeros((n, n))
    indices = np.random.choice(n * n, n, replace=False)
    for index in indices:
        i, j = divmod(index, n)
        matrix[i, j] = matrix[j, i] = np.random.random() * 4 - 2
    
    matrix -= np.identity(n) * np.trace(matrix) / n
    return tk(matrix)