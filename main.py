import numpy as np
import picos
from typing import *
import invariant_implementation as ii
import graph_generate as gg
import matrix_manip as mm
import subspace as ss
from invariant_implementation import *

# No I've never heard of for loops/lists
k1 = c1 = l1 = i1 = gg.complete(1)
k2 = c2 = l2 = gg.complete(2); i2 = gg.independent(2)
k3 = c3 = gg.complete(3); l3 = gg.line(3); i3 = gg.independent(3)
k4 = gg.complete(4); c4 = gg.cycle(4); l4 = gg.line(4); i4 = gg.independent(4)
k5 = gg.complete(5); c5 = gg.cycle(5); l5 = gg.line(5); i5 = gg.independent(5)
k6 = gg.complete(6); c6 = gg.cycle(6); l6 = gg.line(6); i6 = gg.independent(6)
k7 = gg.complete(7); c7 = gg.cycle(7); l7 = gg.line(7); i7 = gg.independent(7)
k8 = gg.complete(8); c8 = gg.cycle(8); l8 = gg.line(8); i8 = gg.independent(8)
k9 = gg.complete(9); c9 = gg.cycle(9); l9 = gg.line(9); i9 = gg.independent(9)

y = gg.from_edge_list(4, (0, 1), (0, 2), (0, 3))
g = gg.from_edge_list(4, (0, 1), (0, 2), (1, 2), (2, 3))
l2sg = gg.from_edge_list(4, (2, 3))
l2sg_alt = gg.from_edge_list(4, (1, 2))

lov_theta_indcp(c5)

lam_1, X_1 = lov_theta_relative(g, l2sg)
lam_2, X_2 = lov_theta_relative(g, l2sg_alt)

print(round(lam_1 - lam_2, 3) == 0) # Different!!
l2s3 = gg.from_edge_list(3, (0, 1))
