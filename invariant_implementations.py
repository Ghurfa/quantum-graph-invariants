from typing import *

import invariant_implementations_cvxpy
import subspace as ss
from matrix_manip import SimpleSymmMatrix, SimpleChoiMatrix
from graph_generate import Graph
from subspace import Subspace
# import invariant_implementations_picos


def lt_general(subspace: Subspace) -> Tuple[float, SimpleSymmMatrix]:
    """
    Calculates min{ max_i{A_ii : A in S, A - J_n is PSD }}

    SDP:
    Minimize t + 1 such that
    1. Y + J_n in S
    2. Y_ii <= t
    3. Y is a PSD n by n matrix
    """

    subspace.ensure_valid()
    return invariant_implementations_cvxpy.lt_general(subspace)

def ind_cp(s1: Subspace, s2: Subspace) -> Tuple[float, SimpleChoiMatrix]:
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
    
    return invariant_implementations_cvxpy.ind_cp(s1, s2)

def lt_quantum(subspace: Subspace) -> Tuple[float, SimpleChoiMatrix]:
    """
    Computes Quantum Lovasz Theta (Duan et al.) of the given subspace

    SDP:
    Maximize lam such that
    1. (id (x) tr)(X) = (1 - lam)(I_n)
    2. X + lam * delta_matrix_n \in (S1 (x) S2) + (S1^perp (x) M_n)
    3. X is a PSD n^2 by n^2 matrix

    SDP taken from prop 4.8 of Araiza et al. As noted there, this is the same as
    ind_cp except for swapping id and tr in constraint 1
    """
    
    subspace.ensure_valid()
    return invariant_implementations_cvxpy.lt_quantum(subspace)

def lt(graph: Graph) -> Tuple[float, SimpleSymmMatrix]:
    return lt_general(ss.eg(graph))

def lt_indcp(graph: Graph) -> Tuple[float, SimpleSymmMatrix]:
    return ind_cp(ss.mn(graph.n), ss.sg(graph))

def lt_relative(gamma: Graph, lam: Graph) -> Tuple[float, SimpleSymmMatrix]:
    return ind_cp(ss.sg(gamma), ss.sg(lam))