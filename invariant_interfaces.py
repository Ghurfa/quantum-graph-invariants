from typing import *

import invariant_implementations as impl
import subspace as ss
from matrix_manip import SimpleSymmMatrix, SimpleChoiMatrix
from graph_generate import Graph
from subspace import Subspace

result_precision = 3

def lt_general(subspace: Subspace) -> Tuple[float, SimpleSymmMatrix]:
    """
    Calculates min{ max_i{A_ii : A in S, A - J_n is PSD }}
    """

    subspace.ensure_valid()
    val, witness = impl.lt_general(subspace)
    return round(val, result_precision), SimpleSymmMatrix(witness)

def ind_cp(s1: Subspace, s2: Subspace) -> Tuple[float, SimpleChoiMatrix]:
    """
    Computes Ind_CP(S1 : S2) (Araiza et al.) of the given subspaces
    """

    s1.ensure_valid()
    s2.ensure_valid()
    if not(s2.is_subspace_of(s1)):
        raise ValueError("S2 is not a sub-operator system of S1")
    
    val, witness = impl.ags_4_1(s1, s2, 0)
    return round(val, result_precision), SimpleChoiMatrix(witness)

def lam_tilde(subspace: Subspace) -> Tuple[float, SimpleChoiMatrix]:
    return f_invar(24, subspace)

def qlt(subspace: Subspace) -> Tuple[float, SimpleChoiMatrix]:
    """
    Computes Quantum Lovasz Theta (Duan et al.) of the given subspace
    """

    subspace.ensure_valid()
    val, witness = impl.ags_4_1(ss.mn(subspace.n), subspace, 1)
    return round(val, result_precision), SimpleChoiMatrix(witness)

def qlt_relative(s1: Subspace, s2: Subspace) -> Tuple[float, SimpleChoiMatrix]:
    """
    Computes relative quantum Lovasz Theta (a la the AGS invariant)
    """

    s1.ensure_valid()
    s2.ensure_valid()
    if not(s2.is_subspace_of(s1)):
        raise ValueError("S2 is not a sub-operator system of S1")
    
    val, witness = impl.ags_4_1(s1, s2, 1)
    return round(val, result_precision), SimpleChoiMatrix(witness)

def lt(graph: Graph) -> Tuple[float, SimpleSymmMatrix]:
    return lt_general(ss.eg(graph))

def lt_relative(gamma: Graph, lam: Graph) -> Tuple[float, SimpleSymmMatrix]:
    return lt_general(ss.sg((gamma - lam).compl))

def quantil(gamma: Graph) -> Tuple[float, SimpleChoiMatrix]:
    return qlt(ss.antilaplacian(gamma))

def lam_til_dsw(subspace: Subspace) -> Tuple[float, SimpleChoiMatrix]:
    return qlt_relative(subspace.compl, ss.ci(subspace.n))

def f_invar(code: int, subspace: Subspace) -> Tuple[float, SimpleChoiMatrix]:
    subspace.ensure_valid()
    assert type(code) == int
    assert code >= 0 and code < 64

    val, witness = impl.f_invar(code)(subspace)
    return round(val, result_precision), SimpleChoiMatrix(witness)