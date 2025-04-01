import numpy as np
from typing import *

import invariant_implementations as ii
import graph_generate as gg
import matrix_manip as mm
import subspace as ss

from invariant_implementations import *

a = np.identity(2).astype(int)
b = np.array([[0, 1], [-1, 0]])
c = np.array([[0, 1], [1, 0]])
d = np.array([[1, 0], [0, -1]])
s1 = Subspace(2); s1.basis = [a, b]; s1.constraints = [c, d]
s2 = Subspace(2); s2.basis = [a]; s2.constraints = [b, c, d]
s1pps2 = Subspace(2); s1pps2.basis = [a, c, d]; s1pps2.constraints = [b]

indcp, X = ind_cp(s1, s2)
lov, Y = lt_general(s1pps2)
print(indcp, lov, X, Y, sep='\n')
