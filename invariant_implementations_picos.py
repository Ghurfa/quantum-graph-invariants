import numpy as np
import picos
from typing import *
import matrix_manip as mm
from matrix_manip import SimpleSymmMatrix, SimpleChoiMatrix
from subspace import Subspace

lam_precision = 3

def lt_general(subspace: Subspace) -> Tuple[float, SimpleSymmMatrix]:
    n = subspace.n

    Y = picos.SymmetricVariable("Y", (n, n))
    t = picos.RealVariable("t")

    prob = picos.Problem()
    prob.objective = "min", t

    # Constraint 1
    for constraint in subspace.constraints:
        prob.add_constraint(((Y + np.ones([n, n])) | constraint) == 0)

    # Constraint 2
    for i in range(n):
        prob.add_constraint(Y[i, i] <= t)

    # Constraint 3
    prob.add_constraint(Y >> 0)

    prob.solve(solver="cvxopt")
    return round(1 + t.value, lam_precision), SimpleSymmMatrix(np.array(Y.value))

def ind_cp(s1: Subspace, s2: Subspace) -> Tuple[float, SimpleChoiMatrix]:
    n = s1.n

    X = picos.SymmetricVariable("X", (n*n, n*n))    # This should be a hermitian variable but cvxopt is crashing when you change it
    lam = picos.RealVariable("lambda")

    prob = picos.Problem()
    prob.objective = "max", lam

    # Constraint 1
    prob.add_constraint(X.partial_trace(subsystems=0, dimensions=n) == (1 - lam) * np.identity(n))

    # Constraint 2
    for s1_bvec in s1.basis:
        for s2_p_bvec in s2.constraints:
            constraint = np.kron(s1_bvec, s2_p_bvec)
            prob.add_constraint((X + lam * mm.delta_matrix(n) | constraint) == 0)
    
    # Constraint 3 
    prob.add_constraint(X >> 0)

    prob.solve(solver="cvxopt")
    return round(1 / lam.value, lam_precision), SimpleChoiMatrix(np.array(X.value))

def lt_quantum(subspace: Subspace) -> Tuple[float, SimpleChoiMatrix]:
    n = subspace.n

    X = picos.SymmetricVariable("X", (n*n, n*n))    # This should be a hermitian variable but cvxopt is crashing when you change it
    lam = picos.RealVariable("lambda")

    prob = picos.Problem()
    prob.objective = "max", lam

    # Constraint 1
    prob.add_constraint(X.partial_trace(subsystems=1, dimensions=n) == (1 - lam) * np.identity(n))

    # Constraint 2
    for i in range(n):
        for j in range(n):
            for constraint in subspace.constraints:
                prob.add_constraint(((X + lam * mm.delta_matrix(n))[i * n : (i + 1) * n, j * n : (j + 1) * n] | constraint) == 0)
    
    # Constraint 3 
    prob.add_constraint(X >> 0)

    prob.solve(solver="cvxopt")
    return round(1 / lam.value, lam_precision), SimpleChoiMatrix(np.array(X.value))
