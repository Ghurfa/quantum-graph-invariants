import numpy as np
from typing import *
from subspace import Subspace

import subspace as ss
import graph_generate as gg
import invariants
import invariant_implementations as ii

def verify_extends(qlt: Callable[[Subspace], float]) -> bool:
    """
    Verify that the Lovasz Theta quantum extension matches the normal Lovasz Theta on graph systems
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
    n1 = 3
    n2 = 4
    for i in range(0, 20):
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

def verify_maxive(qlt: Callable[[Subspace], float]) -> bool:
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

    if verify_maxive(qlt):
        print(f"{name} seems max-ive")
    else:
        print(f"{name} is not max-ive")
    
    exponent = np.log(qlt(ss.ci(7))) / np.log(7)
    print(f"Exponent of {name} is {round(exponent, 3)}")

np.random.seed(10700)
verify_lt_extension(lambda x: ii.lt_general(x)[0], "LT1")
# verify_lt_extension(lambda x: ii.lt2(x)[0], "LT2")
verify_lt_extension(lambda x: ii.araiza_4_1(ss.mn(x.n), x, 0)[0], "Ind_CP")
verify_lt_extension(lambda x: ii.araiza_4_1(ss.mn(x.n), x, 1)[0], "QLT")
