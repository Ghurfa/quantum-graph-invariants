#Parts of this script use code written in 2013 by Daniel Stahlke (https://gist.github.com/dstahlke/6895643)
#The purpose of this file is to compare our values of ind_cp(M_n : S_G) and QLT(S_G) vs a verified formulation of the lovasz theta

from __future__ import print_function

import numpy as np
import cvxopt.base
import cvxopt.solvers

from graph_generate import Graph as CustomGraph
from subspace import Subspace, mn
from invariants import lt_quantum, ind_cp
from subspace import sg
from graph_generate import random


def parse_graph(G, complement=False):
    '''
    Takes a Sage graph, networkx graph, or adjacency matrix as argument, and returns
    vertex count and edge list for the graph and its complement.
    '''
    
    if isinstance(G, CustomGraph):
        
        adj_matrix = G.adjacency_matrix
        G = adj_matrix 
    else:
        
        if type(G).__module__+'.'+type(G).__name__ == 'networkx.classes.graph.Graph':
            import networkx
            G = networkx.convert_node_labels_to_integers(G)
            nv = len(G)
            edges = [(i, j) for (i, j) in G.edges() if i != j]
            c_edges = [(i, j) for (i, j) in networkx.complement(G).edges() if i != j]
        elif type(G).__module__+'.'+type(G).__name__ == 'sage.graphs.graph.Graph':
            G = G.adjacency_matrix().numpy()
            nv = G.shape[0]
            edges = [(j, i) for i in range(nv) for j in range(i) if G[i, j]]
            c_edges = [(j, i) for i in range(nv) for j in range(i) if not G[i, j]]
        else:
            G = np.array(G)

    if isinstance(G, np.ndarray):
        nv = G.shape[0]
        edges = [(j, i) for i in range(nv) for j in range(i) if G[i, j]]
        c_edges = [(j, i) for i in range(nv) for j in range(i) if not G[i, j]]

    for (i, j) in edges:
        assert i < j
    for (i, j) in c_edges:
        assert i < j

    if complement:
        (edges, c_edges) = (c_edges, edges)

    return (nv, edges, c_edges)


def lovasz_theta(G, long_return=False, complement=False):

    (nv, edges, _) = parse_graph(G, complement)
    ne = len(edges)

    if nv == 1:
        return 1.0

    c = cvxopt.matrix([0.0]*ne + [1.0])
    G1 = cvxopt.spmatrix(0, [], [], (nv*nv, ne+1))
    for (k, (i, j)) in enumerate(edges):
        G1[i*nv+j, k] = 1
        G1[j*nv+i, k] = 1
    for i in range(nv):
        G1[i*nv+i, ne] = 1

    G1 = -G1
    h1 = -cvxopt.matrix(1.0, (nv, nv))

    sol = cvxopt.solvers.sdp(c, Gs=[G1], hs=[h1])

    if long_return:
        theta = sol['x'][ne]
        Z = np.array(sol['ss'][0])
        B = np.array(sol['zs'][0])
        return { 'theta': theta, 'Z': Z, 'B': B }
    else:
        return sol['x'][ne]

if __name__ == "__main__":
    ns = [4, 5]
    densities = [0.01, 0.1]
    tol = 1e-4

    for n in ns:
        for d in densities:
            G = random(n, d)

            try:
                theta = lovasz_theta(G)
                S_G = sg(G)

                indcp_val, _ = ind_cp(mn(S_G.n), S_G)
                qtheta_val, _ = lt_quantum(S_G)

                print("Classical Lovász theta:", theta)
                print("Ind_CP(M_n : S_G):", indcp_val)
                print("Quantum Lovász theta:", qtheta_val)

                rel_tol_1 = abs(theta - indcp_val) / abs(indcp_val) <= tol
                rel_tol_2 = abs(indcp_val - qtheta_val) / abs(qtheta_val) <= tol

                if rel_tol_1 and rel_tol_2:
                    print("The values agree")
                else:
                    print("MISMATCH")

            except Exception as e:
                print("Error for n =", n, "density =", d, ":", str(e))
