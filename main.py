import numpy as np
from typing import *

import examples
import invariants as ii
import graph_generate as gg
import matrix_manip as mm
import subspace as ss

from invariants import *
from subspace import Subspace

np.random.seed(10700)
n = 3
for i in range(50):
    quantum_graph, _, _ = ss.random_s1_s2(n)

    indcp, X = ind_cp(ss.mn(n), quantum_graph)
    qlt, Y = lt_quantum(quantum_graph)
    close = np.isclose(indcp, qlt)
    print(n, i, indcp, qlt, np.isclose(indcp, qlt))
    if not close:
        print("Quantum Graph:", quantum_graph, sep='\n')
        print("Ind_CP Value:", indcp)
        print("Ind_CP Witness:", X, sep='\n')
        print("Quantum LT Value:", qlt)
        print("Quantum LT Witness:", Y, sep='\n')
        break