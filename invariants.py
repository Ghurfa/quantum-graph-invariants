from typing import *
import numpy as np

import invariant_implementations as ii
import subspace as ss
from matrix_manip import SimpleSymmMatrix, SimpleChoiMatrix
from graph_generate import Graph
from subspace import Subspace

lam_precision = 15

def lt_general(subspace: Subspace) -> Tuple[float, SimpleSymmMatrix]:
    """
    Calculates min{ max_i{A_ii : A in S, A - J_n is PSD }}
    """

    subspace.ensure_valid()
    val, witness = ii.lt_general(subspace)
    return round(val, lam_precision), SimpleSymmMatrix(witness)

def ind_cp(s1: Subspace, s2: Subspace) -> Tuple[float, SimpleChoiMatrix]:
    """
    Computes Ind_CP(S1 : S2) (Araiza et al.) of the given subspaces
    """

    s1.ensure_valid()
    s2.ensure_valid()
    if not(s2.is_subspace_of(s1)):
        raise ValueError("S2 is not a sub-operator system of S1")
    
    val, witness = ii.araiza_4_1(s1, s2, 0)
    return round(val, lam_precision), SimpleChoiMatrix(witness)

def lam_tilde(subspace: Subspace) -> Tuple[float, SimpleChoiMatrix]:
    subspace.ensure_valid()
    val, witness = ii.araiza_4_1(subspace, ss.ci(subspace.n), 0)
    return round(val, lam_precision), SimpleChoiMatrix(witness)

def lt_quantum(subspace: Subspace) -> Tuple[float, SimpleChoiMatrix]:
    """
    Computes Quantum Lovasz Theta (Duan et al.) of the given subspace
    """

    subspace.ensure_valid()
    val, witness = ii.araiza_4_1(ss.mn(subspace.n), subspace, 1)
    return round(val, lam_precision), SimpleChoiMatrix(witness)

def lt(graph: Graph) -> Tuple[float, SimpleSymmMatrix]:
    return lt_general(ss.eg(graph))

def lt_relative(gamma: Graph, lam: Graph) -> Tuple[float, SimpleSymmMatrix]:
    return lt_general(ss.sg((gamma - lam).compl))

def quantil(gamma: Graph) -> Tuple[float, SimpleChoiMatrix]:
    return lt_quantum(ss.antilaplacian(gamma))
