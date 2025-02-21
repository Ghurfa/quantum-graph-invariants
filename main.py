import numpy as np
import picos
from typing import *
import graph_generate as gg

def lovasz_theta(graph: Dict[int, List[int]]):
    """ Calculate the Lovasz Theta number of a graph using SDP. Also returns the witness

    min t such that
    1. y_ij = -1 if not (i ~ j)
    2. y_ii = t - 1 for i from 1 to n
    3. Y is PSD

    Based on the SDP given in Theorem 3.6.1 in "Approximation Algorithms and Semidefinite Programming" by G\"artner and Matousek
    """

    n = len(graph)
    _, edge_compl = gg.edges(graph)

    Y = picos.SymmetricVariable("Y", (n, n))
    t = picos.RealVariable("t")

    prob = picos.Problem()
    prob.objective = "min", t
    for x in edge_compl:
        prob.add_constraint(Y[x[0], x[1]] == -1)    # Constraint 1
    for i in range(n):
        prob.add_constraint(Y[i, i] == t - 1)       # Constraint 2
    prob.add_constraint(Y >> 0)                     # Constraint 3

    prob.solve(solver="cvxopt")
    return t.value, Y.value

def main():
    lt, sol = lovasz_theta(gg.line(3))
    print(lt)
    print(sol)


if __name__ == "__main__":
    main()
    