import numpy as np
import picos
from typing import *
import graph_generate as gg

def lovasz_theta(graph: Dict[int, List[int]]):
    """
    Calculate the Lovasz Theta number of a graph using SDP. Also returns the witness

    SDP:
    Minimize t such that
    1. y_ij = -1        if not (i ~ j)
    2. y_ii = t - 1     for i from 1 to n
    3. Y is PSD

    SDP taken from Theorem 3.6.1 in "Approximation Algorithms and Semidefinite Programming" by G\"artner and Matousek
    """

    n = len(graph)
    _, non_edges = gg.edges(graph)

    Y = picos.SymmetricVariable("Y", (n, n))
    t = picos.RealVariable("t")

    prob = picos.Problem()
    prob.objective = "min", t
    for x in non_edges:
        prob.add_constraint(Y[x[0], x[1]] == -1)    # Constraint 1
    for i in range(n):
        prob.add_constraint(Y[i, i] == t - 1)       # Constraint 2
    prob.add_constraint(Y >> 0)                     # Constraint 3

    prob.solve(solver="cvxopt")
    return t.value, Y.value

def ind_mn_to_sg(graph: Dict[int, List[int]]):
    """
    Computes Ind_CP(M_n : S_G), where S_G is the graph system corresponding to the input graph, which equals the Lovasz Theta of the graph

    SDP:
    Maximize lam such that
    1. (tr (x) id)(X) = (1 - lam)I_n
    2. X + lam * delta_matrix_n \in (M_n (x) S_G)
    3. X is a PSD n^2 by n^2 matrix

    SDP taken from Proposition 4.1 in "An Index for Inclusions of Operator Systems" by Arazia, Griffin, and Sinclair
    Prop 4.4 proved that this quantity (Ind_CP(M_n : S_G)) equals the Lovasz theta of the graph
    """

    n = len(graph)
    _, non_edges = gg.edges(graph)

    X = picos.SymmetricVariable("X", (n*n, n*n))    # This should be a hermitian variable but cvxopt is crashing when you change it
    lam = picos.RealVariable("lambda")

    prob = picos.Problem()
    prob.objective = "max", lam

    # Constraint 1
    prob.add_constraint(X.partial_trace(subsystems=-1, dimensions=n) == (1 - lam) * np.identity(n))

    # Constraint 2
    for i in range(0, n):
        for j in range(0, n):
            for (edge_i, edge_j) in non_edges:
                if (edge_i, edge_j) == (i, j):
                    prob.add_constraint(X[i * n + edge_i, j * n + edge_j] + lam == 0)
                else:
                    prob.add_constraint(X[i * n + edge_i, j * n + edge_j] == 0)
    
    # Constraint 3 
    prob.add_constraint(X >> 0)

    prob.solve(solver="cvxopt")
    return 1/lam.value


def main():
    print(f"Normal Lovasz Theta of L3: {lovasz_theta(gg.line(3))}")

    for i in range(2, 6):
        print(f"IndCP Lovasz Theta of L{i}: {ind_mn_to_sg(gg.line(i))}")
        
    print(f"IndCP Lovasz Theta of C5: {ind_mn_to_sg(gg.cycle(5))}")

if __name__ == "__main__":
    main()
    