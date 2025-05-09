import numpy as np
from typing import *

import experiments
import invariants as ii
import graph_generate as gg
import matrix_manip as mm
import subspace as ss

from invariants import *
from subspace import Subspace

experiments.antilaplacian_indcp_eq_2()
print(
    lt(gg.petersen()),
    quantil(gg.petersen()),
    sep='\n'
)