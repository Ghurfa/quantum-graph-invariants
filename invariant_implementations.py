import numpy as np
import picos
from typing import *
import graph_generate as gg
from graph_generate import Graph
import matrix_manip as mm
from matrix_manip import SimpleMatrix, SimpleSymmMatrix, SimpleChoiMatrix
from subspace import Subspace
import subspace as ss

lam_precision = 4

def lt_general(subspace: Subspace) -> Tuple[float, SimpleSymmMatrix]:
    """
    Calculates min{ max_i{A_ii : A in S, A - J_n is PSD }}

    SDP:
    Minimize t + 1 such that
    1. Y + J_n in S
    2. Y_ii <= t
    3. Y is a PSD n by n matrix
    """

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

def lt(graph: Graph) -> Tuple[float, SimpleSymmMatrix]:
    """
    Calculate the Lovasz Theta number of a graph. Also returns the witness

    SDP:
    Minimize t such that
    1. y_ij = -1        if not (i ~ j)
    2. y_ii = t - 1     for i from 1 to n
    3. Y is a PSD n by n matrix

    SDP taken from Theorem 3.6.1 in "Approximation Algorithms and Semidefinite Programming" by G\"artner and Matousek
    """

    return lt_general(ss.eg(graph))

def ind_cp(s1: Subspace, s2: Subspace) -> Tuple[float, SimpleSymmMatrix]:
    """
    Computes Ind_CP(S1 : S2) 

    SDP:
    Maximize lam such that
    1. (tr (x) id)(X) = (1 - lam)(I_n)
    2. X + lam * delta_matrix_n \in (S1 (x) S2) + (S1^perp (x) M_n)
    3. X is a PSD n^2 by n^2 matrix

    SDP modified from the one given in prop 4.1
    """
    
    s1.ensure_valid()
    s2.ensure_valid()
    if not(s2.is_subspace_of(s1)):
        raise ValueError("S2 is not a sub-operator system of S1")
    
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

def lt_indcp(graph: Graph) -> Tuple[float, SimpleSymmMatrix]:
    return ind_cp(ss.mn(graph.n), ss.sg(graph))

def lt_relative(gamma: Graph, lam: Graph) -> Tuple[float, SimpleSymmMatrix]:
    return ind_cp(ss.sg(gamma), ss.sg(lam))
