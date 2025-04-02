import numpy as np
from typing import *

import invariant_implementations as ii
import graph_generate as gg
import matrix_manip as mm
import subspace as ss

from invariant_implementations import *
from subspace import Subspace

np.random.seed(10700)
n = 3
for i in range(0, 50):
    quantum_graph, _, _ = ss.random_s1_s2(n)

    indcp, X = ind_cp(ss.mn(n), quantum_graph)
    qlt, Y = lt_quantum(quantum_graph)
    print(i, indcp, qlt, indcp == qlt)
