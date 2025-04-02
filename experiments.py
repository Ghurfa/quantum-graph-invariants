import numpy as np

import graph_generate as gg
import subspace as ss
from invariants import *
from subspace import Subspace

def general_compression_counterexample():
    a = np.identity(2).astype(int)
    b = np.array([[0, 1],
                  [-1, 0]])
    c = np.array([[0, 1], 
                  [1, 0]])
    d = np.array([[1, 0], 
                  [0, -1]])
    s1 = Subspace(2);       s1.basis = [a, b];          s1.constraints = [c, d]
    s2 = Subspace(2);       s2.basis = [a];             s2.constraints = [b, c, d]
    s1pps2 = Subspace(2);   s1pps2.basis = [a, c, d];   s1pps2.constraints = [b]

    indcp, X = ind_cp(s1, s2)
    lov, Y = lt_general(s1pps2)
    print(indcp, lov, X, Y, sep='\n')

def indcp_qlt_equality():
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
