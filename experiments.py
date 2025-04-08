import numpy as np
import random

import graph_generate as gg
import subspace as ss
from invariants import *
from subspace import Subspace

def general_compression_counterexample():
    """
    Counterexample to the conjecture Ind_CP(S1 : S2) = GLT(S1^perp + S2)
    as well as to the relaxed version Ind_CP(M_n : S) = GLT(S)
    """

    quantum_graph = ss.from_constraints([np.array([[1, 0], [0, -1]])])
    indcp, X = ind_cp(ss.mn(2), quantum_graph)
    lov, Y = lt_general(quantum_graph)
    print(indcp, lov, X, Y, sep='\n')

def indcp_qlt_equality():
    """
    Experiment to see if Ind_CP and QLT agree for arbitary matricial systems
    """
    np.random.seed(10700)
    for n in range(2, 10):
        for i in range(50):
            quantum_graph, _, _ = ss.random_s1_s2(n)

            indcp, X = ind_cp(ss.mn(n), quantum_graph)
            qlt, Y = lt_quantum(quantum_graph)
            close = np.isclose(indcp, qlt)
            print(n, i, indcp, qlt, np.isclose(indcp, qlt))
            if not close:
                print("Quantum Graph:", quantum_graph, sep='\n')
                print("Ind_CP Value:", indcp)
                print("Ind_CP Witness:", X, sep='\n')
                print("Quantum LT Value:", qlt)
                print("Quantum LT Witness:", Y, sep='\n')
                break

def indcp_qlt_equality_counterexample():
    """
    Counterexample to the conjecture Ind_CP(M_n : S) = QLT(S)
    """
    # quantum_graph = ss.from_constraints([np.array([
    #     [2, -1, 0],
    #     [-1, -1, 0],
    #     [0, 0 ,-1]])])
    quantum_graph = ss.from_constraints([np.array([
        [2, -3, -3],
        [-3, -1, 0],
        [-3, 0 ,-1]])])

    indcp, X = ind_cp(ss.mn(3), quantum_graph)
    qlt, Y = lt_quantum(quantum_graph)

    print("Quantum Graph:", quantum_graph, sep='\n')
    print("Ind_CP Value:", indcp)
    print("Ind_CP Witness:", X, sep='\n')
    print("Quantum LT Value:", qlt)
    print("Quantum LT Witness:", Y, sep='\n')

def antilaplacian_indcp_eq_2():
    """
    Experiment to see if Ind_CP(M_n : L_G) = 2
    """

    for n in range(1, 12):
        for i in range(min(50, n * n)):
            graph = gg.random(n, 0.5)
            
            if all(len(graph.adj_list[i]) == 0 for i in range(n)):
                continue

            quantum_graph = ss.antilaplacian(graph)

            indcp, X = ind_cp(ss.mn(n), quantum_graph)
            qlt, Y = lt_quantum(quantum_graph)
            
            close2 = np.isclose(2, indcp, atol=0.001)
            print(n, i, indcp, qlt, close2)
            if not close2:
                print("Graph:", graph, sep='\n')
                print("Quantum Graph:", quantum_graph, sep='\n')
                print("Ind_CP Value:", indcp)
                print("Ind_CP Witness:", X, sep='\n')
                print("Quantum LT Value:", qlt)
                print("Quantum LT Witness:", Y, sep='\n')
                return

def quantil_vs_n():
    """
    Experiment to how QLT(L_G) varies with the number of vertices of G
    """

    for n in range(1, 12):
        for i in range(min(50, n * n)):
            graph = gg.random(n, 0.1)
            
            if all(len(graph.adj_list[i]) == 0 for i in range(n)):
                continue

            qlt, Y = quantil(graph)
            
            num_edges = sum(len(graph.adj_list[k]) for k in range(n)) // 2
            possible_edges = (n * n - n) // 2
            density = num_edges / possible_edges

            print(n, i, qlt, density, 1/density)
            
def quantil_vs_density():
    """
    Experiment to how QLT(L_G) varies with the density of G
    """
    
    n = 8
    vals = []
    for i in range(11):
        goal_density = i / 10
        total = 0
        for _ in range(0, 30):
            graph = gg.random(n, goal_density)
            
            if all(len(graph.adj_list[i]) == 0 for i in range(n)):
                continue

            qlt, Y = quantil(graph)
            
            num_edges = sum(len(graph.adj_list[k]) for k in range(n)) // 2
            possible_edges = (n * n - n) // 2
            density = num_edges / possible_edges

            print(goal_density, qlt, density, 1 / density, sep='\t')
            total += qlt
        vals.append(total)        
    print(vals)

def quantaj():
    for n in range(2, 12):
        for i in range(min(50, n * n)):
            graph = gg.random(n, random.random())
            
            if all(len(graph.adj_list[i]) == 0 for i in range(n)):
                continue
            
            qg = ss.from_constraints([graph.adjacency_matrix[0]])
            indcp, X = ind_cp(ss.mn(n), qg)
            qlt, Y = lt_quantum(qg)
          
            print(n, i, indcp, qlt)

# np.random.seed(10700)
# # indcp_qlt_equality()
# indcp_infeasibility()