import numpy as np
import picos
from typing import *
import graph_generate as gg
from graph_generate import Graph

def lovasz_theta(graph: Graph):
    """
    Calculate the Lovasz Theta number of a graph using SDP. Also returns the witness

    SDP:
    Minimize t such that
    1. y_ij = -1        if not (i ~ j)
    2. y_ii = t - 1     for i from 1 to n
    3. Y is a PSD n by n matrix

    SDP taken from Theorem 3.6.1 in "Approximation Algorithms and Semidefinite Programming" by G\"artner and Matousek
    """

    n = graph.n()
    _, non_edges = graph.edges()

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

def ind_mn_to_sg(graph: Graph):
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

    n = graph.n()
    _, non_edges = graph.edges()

    X = picos.SymmetricVariable("X", (n*n, n*n))    # This should be a hermitian variable but cvxopt is crashing when you change it
    lam = picos.RealVariable("lambda")

    prob = picos.Problem()
    prob.objective = "max", lam

    # Constraint 1
    prob.add_constraint(X.partial_trace(subsystems=0, dimensions=n) == (1 - lam) * np.identity(n))

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
    return 1 / lam.value

def ind_sg_to_sl(graph_g: Graph, graph_l: Graph) -> float:
    """
    Computes Ind_CP(S_G : S_L), where L is a subgraph of G and S_G and S_L are the graph systems corresponding to G and L, respectively

    If graph_l has fewer vertices than graph_g, then it is "padded" with extra vertexes

    This is the "relative Lovasz Theta" defined by Arazia, Griffin, and Sinclair.

    SDP:
    Maximize lam such that
    1. (tr (x) id)(X) = (1 - lam)(I_n)
    2. X + lam * delta_matrix_n \in (S_G (x) S_L) + (S_G^perp (x) M_n)
    3. X is a PSD n^2 by n^2 matrix

    SDP modified from the one given in prop 4.1.
    """

    n = graph_g.n()

    _, non_edges_g = graph_g.edges()
    _, non_edges_l = graph_l.edges()

    X = picos.SymmetricVariable("X", (n*n, n*n))    # This should be a hermitian variable but cvxopt is crashing when you change it
    lam = picos.RealVariable("lambda")

    prob = picos.Problem()
    prob.objective = "max", lam

    # Constraint 1
    prob.add_constraint(X.partial_trace(subsystems=0, dimensions=n) == (1 - lam) * np.identity(n))

    # Constraint 2
    for i in range(0, n):
        for j in range(0, n):
            if (i, j) in non_edges_g:
                continue
            for (edge_i, edge_j) in non_edges_l:
                if (edge_i, edge_j) == (i, j):
                    prob.add_constraint(X[i * n + edge_i, j * n + edge_j] + lam == 0)
                else:
                    prob.add_constraint(X[i * n + edge_i, j * n + edge_j] == 0)
    
    # Constraint 3 
    prob.add_constraint(X >> 0)

    prob.solve(solver="cvxopt")
    return 1 / lam.value, X.value

relative_lov_theta = ind_sg_to_sl

def pretty_print(matrix, precision: int = 2):
    width, height = matrix.size
    for i in range(0, height):
        for j in range(0, width):
            val = 0 if (abs(matrix[i, j]) < 0.01) else matrix[i, j]
            print(f"{val:.{precision}f}".rjust(6), end='')
        print()

def main():
    # print(f"Normal Lovasz Theta of L3: {lovasz_theta(gg.line(3))}")

    # for i in range(2, 6):
    #     print(f"IndCP Lovasz Theta of L{i}: {ind_mn_to_sg(gg.line(i))}")
        
    # print(f"IndCP Lovasz Theta of C5: {ind_mn_to_sg(gg.cycle(5))}")

    # print(f"Lovasz Theta of K5 : C5: {relative_lov_theta(gg.complete(5), gg.cycle(5))}")

    for n in range(3, 6):
        L_big = gg.line(n)
        L2_sub_L_big = Graph.from_edge_list(n, (0, 1))
        val, matrix = relative_lov_theta(L_big, L2_sub_L_big)
        print(f"L2 : L{n} = {val}")
        # pretty_print(matrix)
    
    g = Graph.from_edge_list(4, (0, 1), (1, 2), (1, 2), (2, 3))
    L2_sub_g = Graph.from_edge_list(4, (0, 1))
    L2_to_g, matrix = relative_lov_theta(g, L2_sub_g)
    print(f"L2 : G = {L2_to_g}")

if __name__ == "__main__":
    main()
    