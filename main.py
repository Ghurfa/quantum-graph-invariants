import numpy as np
import picos
from typing import *
import graph_generate as gg
from graph_generate import Graph
import matrix_manip as mm
from matrix_manip import SimpleMatrix, SimpleSymmMatrix, SimpleChoiMatrix

lam_precision = 4

def lovasz_theta(graph: Graph) -> Tuple[float, SimpleSymmMatrix]:
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
    return round(t.value, lam_precision), SimpleSymmMatrix(np.matrix(Y.value))

def ind_mn_to_sg(graph: Graph) -> Tuple[float, SimpleSymmMatrix]:
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
    for i in range(n):
        for j in range(n):
            for (edge_i, edge_j) in non_edges:
                if (edge_i, edge_j) == (i, j):
                    prob.add_constraint(X[i * n + edge_i, j * n + edge_j] + lam == 0)
                else:
                    prob.add_constraint(X[i * n + edge_i, j * n + edge_j] == 0)
    
    # Constraint 3 
    prob.add_constraint(X >> 0)

    prob.solve(solver="cvxopt")
    return round(1 / lam.value, lam_precision), SimpleChoiMatrix(np.matrix(X.value))

def ind_sg_to_sl(graph_g: Graph, graph_l: Graph) -> Tuple[float, SimpleSymmMatrix]:
    """
    Computes Ind_CP(S_G : S_L), where L is a subgraph of G and S_G and S_L are the graph systems corresponding to G and L, respectively

    This is the "relative Lovasz Theta" defined by Arazia, Griffin, and Sinclair.

    SDP:
    Maximize lam such that
    1. (tr (x) id)(X) = (1 - lam)(I_n)
    2. X + lam * delta_matrix_n \in (S_G (x) S_L) + (S_G^perp (x) M_n)
    3. X is a PSD n^2 by n^2 matrix

    SDP modified from the one given in prop 4.1.
    """

    if not(graph_l.is_subgraph_of(graph_g)):
        raise ValueError("graph_l is not a subgraph of graph_g")
    
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
    for i in range(n):
        for j in range(n):
            if any(map(lambda x: x[0] == i and x[1] == j, non_edges_g)):
                continue
            for (edge_i, edge_j) in non_edges_l:
                if (edge_i, edge_j) == (i, j):
                    prob.add_constraint(X[i * n + edge_i, j * n + edge_j] + lam == 0)
                else:
                    prob.add_constraint(X[i * n + edge_i, j * n + edge_j] == 0)
    
    # Constraint 3 
    prob.add_constraint(X >> 0)

    prob.solve(solver="cvxopt")
    return round(1 / lam.value, lam_precision), SimpleChoiMatrix(np.matrix(X.value))

relative_lov_theta = ind_sg_to_sl

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

lam_1, X_1 = relative_lov_theta(g, l2sg)
lam_2, X_2 = relative_lov_theta(g, l2sg_alt)

print(round(lam_1 - lam_2, 3) == 0) # Different!!
