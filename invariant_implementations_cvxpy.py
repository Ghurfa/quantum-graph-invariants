import cvxpy
import numpy as np
from typing import *

import matrix_manip as mm
from matrix_manip import SimpleSymmMatrix, SimpleChoiMatrix
from subspace import Subspace

lam_precision = 3

def lt_general(subspace: Subspace) -> Tuple[float, SimpleSymmMatrix]:
    n = subspace.n

    Y = cvxpy.Variable((n, n), symmetric=True)
    t = cvxpy.Variable(1)

    constraints = [Y >> 0]  # Constraint 3

    # Constraint 1
    for constraint in subspace.constraints:
        constraints.append(cvxpy.trace((Y + np.ones([n, n])) @ constraint) == 0)

    # Constraint 2
    for i in range(n):
        constraints.append(Y[i, i] <= t)

    prob = cvxpy.Problem(cvxpy.Minimize(t), constraints)
    prob.solve()
    return round(1 + float(t.value), lam_precision), SimpleSymmMatrix(np.array(Y.value))

def ind_cp(s1: Subspace, s2: Subspace) -> Tuple[float, SimpleChoiMatrix]:    
    n = s1.n

    X = cvxpy.Variable((n * n, n * n), symmetric=True)
    lam = cvxpy.Variable(1)

    constraints = [
        X >> 0,                                                         # Constraint 1
        cvxpy.partial_trace(X, (n, n), 0) == (1 - lam) * np.identity(n)    # Constraint 3
    ]
    
    # Constraint 2
    for s1_bvec in s1.basis:
        for s2_p_bvec in s2.constraints:
            constraint = np.kron(s1_bvec, s2_p_bvec).conj().T
            constraints.append(cvxpy.trace((X + lam * mm.delta_matrix(n)) @ constraint) == 0)
    
    prob = cvxpy.Problem(cvxpy.Maximize(lam), constraints)
    prob.solve()
    return round(1 / float(lam.value), lam_precision), SimpleChoiMatrix(np.array(X.value))

def lt_quantum(subspace: Subspace) -> Tuple[float, SimpleChoiMatrix]:
    n = subspace.n

    X = cvxpy.Variable((n * n, n * n), symmetric=True)
    lam = cvxpy.Variable(1)

    constraints = [
        X >> 0,                                                         # Constraint 1
        cvxpy.partial_trace(X, (n, n), 0) == (1 - lam) * np.identity(n)    # Constraint 3
    ]

    # Constraint 2
    for i in range(n):
        for j in range(n):
            for s_const in subspace.constraints:
                constraint = np.kron(mm.e_matrix(n, i, j), s_const).conj().T
                constraints.append(cvxpy.trace((X + lam * mm.delta_matrix(n)) @ constraint) == 0)

    prob = cvxpy.Problem(cvxpy.Maximize(lam), constraints)
    prob.solve()
    return round(1 / float(lam.value), lam_precision), SimpleChoiMatrix(np.array(X.value))
