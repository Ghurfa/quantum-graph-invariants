import cvxpy
import numpy as np
from typing import *

import graph_generate as gg
import matrix_manip as mm
import subspace as ss

from invariant_interfaces import *
from subspace import Subspace


a = ss.from_basis([np.identity(2)])
b = ss.from_basis([
    np.identity(2),
    np.array([[1, 1], [1, 0]])
])
c = ss.cotensor(a, b)

print(a)
X1, Y1 = f_invar(7, a)
X2, Y2 = f_invar(7, b)
X3, Y3 = f_invar(7, c)

print(X1)
print(X2)
print(X3)
print(Y1)
print(Y2)
print(Y3)

# print(*lam_tilde(ss.sg(gg.independent(2))), sep='\n')
# print(*lam_tilde(ss.direct_union(ss.sg(gg.independent(2)), ss.sg(gg.independent(2)))), sep='\n')
# a = gg.cycle(5)
# b = gg.from_edge_list(3, (0, 1), (1,2))
# c = gg.other_product_thingy(a, b)

# qgc = ss.cotensor(ss.sg(a), ss.sg(b))
# qgc_2 = ss.sg(c)
# qgc.ensure_valid()
# qgc_2.ensure_valid()

# print("Dim qgc:", len(qgc.s0) + 1)
# print("Dim qgc_2:", len(qgc_2.s0) + 1)
# print("Eq:", qgc.is_subspace_of(qgc_2))

# a, b = impl.lt1_dual(ss.ci(3))
# d = np.array([
#     [0.4 + 1/30, 1/3, -0.2 - 1/30],
#     [1/3, 0.4 + 1/30, -0.2 - 1/30],
#     [-0.2 - 1/30, -0.2 - 1/30, 0.5 + 1/30]
# ])
# d /= np.trace(d)
# c = d.sum((0, 1))
# #c, d = ii.lt1_dual(ss.eg(gg.from_edge_list(3, (0, 1))))
# #c = 1
# #d = np.array([[0.25, 0, 0], [0, 0.25, 0], [0, 0, 0.5]])

# print(a, SimpleSymmMatrix(b), c, SimpleSymmMatrix(d), sep='\n')


# X = cvxpy.Variable((6, 6), symmetric=True)
# lam1 = cvxpy.Variable((1, 1))
# lam2 = cvxpy.Variable((1, 1))

# constraints = [
#     X >> 0,               # Constraint 1
#     X[0:3, 0:3] == b * lam1,
#     X[3:6, 3:6] == d * lam2,
#     cvxpy.trace(X) == 1
# ]

# prob = cvxpy.Problem(cvxpy.Maximize(cvxpy.sum(X)), constraints)
# prob.solve()
# e, f = float(prob.objective.value), X.value
# print(e)
# print(SimpleSymmMatrix(f, 4))