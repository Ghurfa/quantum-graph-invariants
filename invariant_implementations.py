import cvxpy
import numpy as np
from typing import *

import matrix_manip as mm
from matrix_manip import SimpleSymmMatrix, SimpleChoiMatrix
from subspace import Subspace

def lt_general(subspace: Subspace) -> Tuple[float, np.ndarray]:
    """
    Calculates min{ max_i{A_ii : A in S, A - J_n is PSD }}

    SDP:
    Minimize t + 1 such that
    1. Y + J_n in S
    2. Y_ii <= t
    3. Y is a PSD n by n matrix
    """

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
    return 1 + float(t.value), Y.value

def araiza_4_1(s1: Subspace, s2: Subspace, pt_axis: int) -> Tuple[float, np.ndarray]:
    """
    Computes either the SDP in Prop 4.1 (gives Ind_CP(S1 : S2)) or the SDP in Prop 4.8
    (gives quantum Lovasz Theta). They differ only by the axis of the partial trace in constraint 1.
    pt_axis = 0 is Ind_CP and pt_axis = 1 is QLT (I think)

    SDP:
    Maximize lam such that
    1. (tr (x) id)(X) = (1 - lam)(I_n)              or          (id (x) tr)(X) = (1 - lam)(I_n)
    2. X + lam * delta_matrix_n \in (S1 (x) S2) + (S1^perp (x) M_n)
    3. X is a PSD n^2 by n^2 matrix

    SDP modified from the one given in prop 4.1
    """

    n = s1.n

    X = cvxpy.Variable((n * n, n * n), symmetric=True)
    lam = cvxpy.Variable(1)

    constraints = [
        cvxpy.partial_trace(X, (n, n), pt_axis) == (1 - lam) * np.identity(n),  # Constraint 1
        X >> 0                                                                  # Constraint 3
    ]
    
    # Constraint 2
    for s1_bvec in s1.basis:
        for s2_perp_bvec in s2.constraints:
            constraint = np.kron(s1_bvec, s2_perp_bvec).conj().T
            constraints.append(cvxpy.trace((X + lam * mm.delta_matrix(n)) @ constraint) == 0)
    
    prob = cvxpy.Problem(cvxpy.Maximize(lam), constraints)
    prob.solve()
    return 1 / float(lam.value), X.value

