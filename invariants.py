from typing import *

import invariant_implementations
import subspace as ss
from matrix_manip import SimpleSymmMatrix, SimpleChoiMatrix
from graph_generate import Graph
from subspace import Subspace

def lt_general(subspace: Subspace) -> Tuple[float, SimpleSymmMatrix]:
    """
    Calculates min{ max_i{A_ii : A in S, A - J_n is PSD }}
    """

    subspace.ensure_valid()
    return invariant_implementations.lt_general(subspace)

def ind_cp(s1: Subspace, s2: Subspace) -> Tuple[float, SimpleChoiMatrix]:
    """
    Computes Ind_CP(S1 : S2) (Araiza et al.) of the given subspaces
    """

    s1.ensure_valid()
    s2.ensure_valid()
    if not(s2.is_subspace_of(s1)):
        raise ValueError("S2 is not a sub-operator system of S1")
    
    return invariant_implementations.araiza_4_1(s1, s2, 0)

def lt_quantum(subspace: Subspace) -> Tuple[float, SimpleChoiMatrix]:
    """
    Computes Quantum Lovasz Theta (Duan et al.) of the given subspace
    """

    subspace.ensure_valid()
    return invariant_implementations.araiza_4_1(ss.mn(subspace.n), subspace, 1)

def lt(graph: Graph) -> Tuple[float, SimpleSymmMatrix]:
    return lt_general(ss.eg(graph))

def lt_indcp(graph: Graph) -> Tuple[float, SimpleSymmMatrix]:
    return ind_cp(ss.mn(graph.n), ss.sg(graph))

def lt_relative(gamma: Graph, lam: Graph) -> Tuple[float, SimpleSymmMatrix]:
    return ind_cp(ss.sg(gamma), ss.sg(lam))