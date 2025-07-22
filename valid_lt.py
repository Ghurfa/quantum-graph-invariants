import numpy as np
from typing import *
from subspace import Subspace

import graph_generate as gg
import invariants
import invariant_implementations as ii
import matrix_manip as mm
import subspace as ss

def verify_extends(qlt: Callable[[Subspace], float]) -> bool:
    """
    Verify that the Lovasz Theta quantum extension matches the classical Lovasz Theta on graph systems
    """
    
    for n in range(2, 5):
        for i in range(n * n):
            graph = gg.random(n, i / (n * n))
            expected = invariants.lt(graph)[0]
            actual = qlt(ss.sg(graph))
            if not(np.isclose(actual, expected, rtol=0.001)):
                return False
    return True

def verify_multiplicative(qlt: Callable[[Subspace], float]) -> bool:
    n1 = 3
    n2 = 2
    for i in range(0, 20):
        qg1 = ss.random(n1)
        qg2 = ss.random(n2)
        prod = ss.tensor(qg1, qg2)
        
        ltg1 = qlt(qg1)
        ltg2 = qlt(qg2)
        actual = qlt(prod)
        expected = ltg1 * ltg2

        if not(np.isclose(actual, expected, rtol = 0.001)):
            return False
    
    return True

def verify_additive(qlt: Callable[[Subspace], float]) -> bool:
    n1 = 2
    n2 = 3
    for i in range(30):
        qg1 = ss.random(n1)
        qg2 = ss.random(n2)
        dir_sum = ss.direct_sum(qg1, qg2)
        
        ltg1 = qlt(qg1)
        ltg2 = qlt(qg2)
        actual = qlt(dir_sum)
        expected = ltg1 + ltg2

        if not(np.isclose(actual, expected, rtol = 0.001)):
            return False
    
    return True

def verify_maxitive(qlt: Callable[[Subspace], float]) -> bool:
    n1 = 4
    n2 = 4
    for i in range(0, 20):
        qg1 = ss.random(n1)
        qg2 = ss.random(n2)
        comp_sum = ss.complete_sum(qg1, qg2)
        
        ltg1 = qlt(qg1)
        ltg2 = qlt(qg2)
        actual = qlt(comp_sum)
        expected = max(ltg1, ltg2)

        if not(np.isclose(actual, expected, rtol = 0.001)):
            return False
    
    return True

def verify_unitary_invar(qlt: Callable[[Subspace], float]):

    n = 5
    for i in range(0, 5):
        qg = ss.random(n)
        uni = mm.rand_uni(n)

        lt1 = qlt(qg)
        qg2 = ss.from_constraints(ss.get_conjugate_basis(uni, qg.constraints))
        lt2 = qlt(qg2)

        if not(np.isclose(lt1, lt2, rtol=0.001)):
            return False

    return True

def verify_induced_monotonic(qlt: Callable[[Subspace], float]):

    for n in range(3, 6):
        for i in range(0, 5):
            qg = ss.random(n)
            uni = mm.rand_uni(n)
            k = np.random.randint(2, n)
            semi_uni = uni[0 : n, 0 : k]

            lt1 = qlt(qg)
            nice_basis = [mm.SimpleSymmMatrix(x) for x in ss.get_conjugate_basis(semi_uni, qg.basis)]
            new_basis = [np.identity(k)]
            ss.extend_basis(new_basis, iter(ss.get_conjugate_basis(semi_uni, qg.basis)))
            qg2 = ss.from_basis(new_basis)
            lt2 = qlt(qg2)

            if not(np.isclose(lt1, lt2, rtol=0.001) or lt2 <= lt1):
                return False
    
    return True

def verify_lt_extension(qlt: Callable[[Subspace], float], name: str):
    """
    Verify that the Lovasz Theta quantum extension matches the normal Lovasz Theta on graph
    systems and is multiplicative, additive, and max-ive
    """

    if verify_extends(qlt):
        print(f"{name} seems to extends the classical LT")
    else:
        print(f"{name} does not extends the classical LT")
    
    if verify_multiplicative(qlt):
        print(f"{name} seems multiplicative")
    else:
        print(f"{name} is not multiplicative")

    if verify_additive(qlt):
        print(f"{name} seems additive")
    else:
        print(f"{name} is not additive")

    if verify_maxitive(qlt):
        print(f"{name} seems max-ive")
    else:
        print(f"{name} is not max-ive")
    
    if verify_unitary_invar(qlt):
        print(f"{name} seems invariant under conjugation with unitary")
    else:
        print(f"{name} is not invariant under conjugation with unitary")

    if verify_induced_monotonic(qlt):
        print(f"{name} seems monotonic under induced sububgraphs")
    else:
        print(f"{name} is not monotonic under induced sububgraphs")
    
    exponent = np.log(qlt(ss.ci(7))) / np.log(7)
    print(f"Exponent of {name} is {round(exponent, 3)}")

def gen_invar_dual(code):
    def invar(subspace: Subspace) -> float:
        import cvxpy
        """
        some QLT candidate
        """
        n = subspace.n

        X = cvxpy.Variable((n, n), symmetric=True)
        Y = cvxpy.Variable((n * n, n * n), symmetric=True)

        constraints = [
            cvxpy.trace(X) == 1,
            X >> 0,
            cvxpy.kron(np.identity(n), X) + Y >> 0
        ]
        
        basis = [np.identity(n)]
        ss.extend_basis(basis, iter(subspace.basis))
        dim = len(basis)
        ss.extend_basis(basis)

        parts = [basis[dim:], basis[1:dim], basis[:1]]
        for i, first_list in enumerate(parts):
            for j, second_list in enumerate(parts):
                mask = 1 << (3 * i + j)
                should_constrain = (code & mask) == 0
                if not(should_constrain):
                    continue

                for a in first_list:
                    for b in second_list:
                        constraints.append(cvxpy.trace(Y @ np.kron(a, b)) == 0)
        
        prob = cvxpy.Problem(cvxpy.Maximize(cvxpy.trace((cvxpy.kron(np.identity(n), X) + Y) @ mm.delta_matrix(n))), constraints)
        prob.solve()
        return float(prob.objective.value), X.value, Y.value
    return invar

np.random.seed(10702)

# Best code I've ever written
def shuffle_iso(mat: np.ndarray) -> np.ndarray:
    n = int(mat.shape[0] ** 0.25)
    new_mat = np.zeros_like(mat)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    from_x = i * n * n * n + j * n * n + k * n + l
                    to_x = i * n * n * n + k * n * n + j * n + l
                    for ii in range(n):
                        for jj in range(n):
                            for kk in range(n):
                                for ll in range(n):
                                    from_y = ii * n * n * n + jj * n * n + kk * n + ll
                                    to_y = ii * n * n * n + kk * n * n + jj * n + ll
                                    new_mat[to_x, to_y] = mat[from_x, from_y]

    return new_mat

np.random.seed(10700)
verify_lt_extension(ii.f_invar(1), "(S^perp x S^perp)^perp")