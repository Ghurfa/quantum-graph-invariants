import numpy as np
import random
from typing import *

import invariant_implementations as ii
import graph_generate as gg
import matrix_manip as mm
import subspace as ss

from invariant_implementations import *

# # No I've never heard of for loops/lists
# k1 = c1 = l1 = i1 = gg.complete(1)
# k2 = c2 = l2 = gg.complete(2); i2 = gg.independent(2)
# k3 = c3 = gg.complete(3); l3 = gg.line(3); i3 = gg.independent(3)
# k4 = gg.complete(4); c4 = gg.cycle(4); l4 = gg.line(4); i4 = gg.independent(4)
# k5 = gg.complete(5); c5 = gg.cycle(5); l5 = gg.line(5); i5 = gg.independent(5)
# k6 = gg.complete(6); c6 = gg.cycle(6); l6 = gg.line(6); i6 = gg.independent(6)
# k7 = gg.complete(7); c7 = gg.cycle(7); l7 = gg.line(7); i7 = gg.independent(7)
# k8 = gg.complete(8); c8 = gg.cycle(8); l8 = gg.line(8); i8 = gg.independent(8)
# k9 = gg.complete(9); c9 = gg.cycle(9); l9 = gg.line(9); i9 = gg.independent(9)

# y = gg.from_edge_list(4, (0, 1), (0, 2), (0, 3))
# g = gg.from_edge_list(4, (0, 1), (0, 2), (1, 2), (2, 3))
# l2sg = gg.from_edge_list(4, (2, 3))
# l2sg_alt = gg.from_edge_list(4, (1, 2))

# lam_1, X_1 = lt_relative(g, l2sg)
# lam_2, X_2 = lt_relative(g, l2sg_alt)

# print(round(lam_1 - lam_2, 3) == 0) # Different!!
# l2s3 = gg.from_edge_list(3, (0, 1))
    
# s1 = ss.Subspace(2)
# s1.basis = [
#     np.array([[1, -1], [-1, 0]]),
#     np.array([[0, 1], [1, 1]])
# ]
# s1.constraints = [
#     np.array([[0, 1], [-1, 0]]),
#     np.array([[2, 1], [1, -2]])
# ]

# s2 = ss.Subspace(2)
# s2.basis = [np.array([[1, 0], [0, 1]])]
# s2.constraints = [
#     np.array([[1, 0], [0, -1]]),
#     np.array([[0, 1], [0, 0]]),
#     np.array([[0, 0], [1, 0]])
# ]

# s1pps2 = ss.Subspace(2)
# s1pps2.basis = s1.constraints + s2.basis
# s1pps2.constraints = [np.array([[-1, 2], [2, 1]])]

# indcp, X = ind_cp(s1, s2)
# lt, Y = lt_general(s1pps2)
# print(indcp, X, lt, Y, sep='\n')

random.seed(10700)
for i in range(0, 50):
    s1, s2, s1pps2 = ss.random_s1_s2(3)

    try:
        indcp, X = ind_cp(s1, s2)
    except:
        continue    # idk why most are crashing

    lov, Y = lt_general(s1pps2)
    print(i, indcp, lov, indcp == lov)
    if indcp != lov:
        for m in s2.basis:
            print(SimpleMatrix(m, 0))
            print('---------')
        print('-----------------------------------------------------')
        for m in s1pps2.constraints:
            print(SimpleMatrix(m, 0))
            print('---------')
        print('-----------------------------------------------------')
        for m in s1.constraints:
            print(SimpleMatrix(m, 0))
            print('---------')
        break
