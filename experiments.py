import numpy as np
import random

import invariant_implementations as ii
import graph_generate as gg
import matrix_manip as mm
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

            print(n, i, qlt, density, 1 / density)
            
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
    
def quantil_vs_lt():
    """
    Experiment comparing QLT(L_G) to LT(G) where G is a graph and L is its antilaplacian matrix
    """
    for n in range(2, 10):
        for i in range(n * n):
            graph = gg.random(n, 0.1 * i % 10)

            quantil, X = lt_quantum(ss.antilaplacian(graph))
            lovt, Y = lt(graph.compl)

            print(quantil, lovt, np.isclose(quantil, lovt))

def quantaj():
    """
    Experiment with values of QLT(span{A}^perp) where A is the adjacency matrix of the graph
    """

    for n in range(2, 12):
        for i in range(min(50, n * n)):
            graph = gg.random(n, random.random())
            
            if all(len(graph.adj_list[i]) == 0 for i in range(n)):
                continue
            
            qg = ss.from_constraints([graph.adjacency_matrix])
            indcp, X = ind_cp(ss.mn(n), qg)
            qlt, Y = lt_quantum(qg)

            print(n, i, indcp, qlt)

def indcp_codim_1_eq_2():
    """
    Experiment to see if Ind_CP(M_n : T_K) = 2
    """

    for n in range(2, 10):
        for i in range(0, n * n):
            K = np.zeros((n, n))
            while np.array_equal(K, np.zeros((n, n))):
                K = np.round(np.random.rand(n, n) * 10, decimals=0)
                K += K.T
                K -= np.trace(K)/n * np.identity(n)

            indcp, X = ind_cp(ss.mn(n), ss.tk(K))
            print(n, i, indcp, np.isclose(indcp, 2, rtol=0.001), sep='\t')

def non_self_adj_indcp_tk():
    """
    Counterexample to the conjecture that Ind_CP(T_K) = 2 even when K is not Hermitian
    """

    K = np.array([
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0]
    ])

    sub = Subspace(3)
    sub.constraints = [K]
    indcp, Y = ii.araiza_4_1(ss.mn(3), sub, 0)
    close = np.isclose(indcp, 2, rtol=0.001)
    print(indcp, close, sep='\t')

def rand_mat(n) -> np.ndarray:
    matrix = np.zeros((n, n))
    indices = np.random.choice(n * n, n, replace=False)
    for index in indices:
        i, j = divmod(index, n)
        matrix[i, j] = matrix[j, i] = np.random.random() * 4 - 2
    
    return matrix

def lam_tilde_tk():
    for n in range(2, 10):
        for i in range(n * n):
            mat = rand_mat(n)
            mat -= np.trace(mat) * np.identity(n) / n
            sub = ss.from_constraints([mat])
            lam_til, Y = lam_tilde(sub)
            print(n, i, lam_til, sep='\t')

def indcp_lamtil_tks_eq_n():
    """
    Experiment checking the conjecture Ind_CP(M_n : T_K*) = Ind_CP(T_K* : CI) = n
    """

    for n in range(2, 10):
        for i in range(n * n):
            K = rand_mat(n)
            K -= np.trace(K) * np.identity(n) / n
            sub_b = ss.tks(K)
            indcp, X = ind_cp(ss.mn(n), sub_b)
            lam_til_b, Z = lam_tilde(sub_b)
            good = np.isclose(indcp, n, rtol=0.001) and np.isclose(lam_til_b, n, rtol=0.001)
            print(n, i, indcp, lam_til_b, good, sep='\t')

def s_star():
    """
    Experiment looking at values of random diagonal-restricted subspaces relative to M_n and CI
    for both Ind_CP and QLT
    """

    for n in range(2, 10):
        for i in range(n * n):
            sub = ss.diag_restricted(ss.random(n))
            indcp, X = ind_cp(ss.mn(n), sub)
            lam_til, Y = lam_tilde(sub)
            qlt, Z = lt_quantum(sub)
            qlt2, W = ii.araiza_4_1(sub, ss.ci(n), 1)
            ordering = [b for (a, b) in sorted(zip([indcp, lam_til, qlt, qlt2], range(4)))]
            
            print(n, i, indcp, lam_til, qlt, qlt2, str(ordering), sep='\t')

def dot_score(choi_mat: SimpleChoiMatrix, mat: np.ndarray) -> SimpleSymmMatrix:
    n = mat.shape[0]
    ret = np.zeros(mat.shape)
    for i in range(n):
        for j in range(n):
            slice = choi_mat[i * n : (i + 1) * n, j * n : (j + 1) * n]
            ret[i, j] = np.trace(slice @ mat.conj().T)
    return SimpleSymmMatrix(ret)

def cofactors(A):
    U,sigma,Vt = np.linalg.svd(A)
    N = len(sigma)
    g = np.tile(sigma,N)
    g[::(N+1)] = 1
    G = np.diag(-(-1)**N*np.prod(np.reshape(g,(N,N)),1)) 
    return U @ G @ Vt 

def remainder_component(choi_mat: SimpleChoiMatrix, K: np.ndarray) -> Tuple[SimpleChoiMatrix, np.ndarray]:
    n = K.shape[0]
    ret = np.zeros((n * n, n * n))
    removed = np.zeros((n * n, n * n))
    for i in range(n):
        for j in range(n):
            slice = choi_mat[i * n : (i + 1) * n, j * n : (j + 1) * n]
            i_comp = np.trace(slice) / n * np.identity(n)
            k_comp = np.trace(slice @ K.conj().T) / np.trace(K @ K.conj().T) * K
            k2_comp = np.trace(slice @ K @ K) / np.trace(K @ K @ K @ K) * (K @ K)
            e = mm.e_matrix(n, i, j)
            ret += np.kron(slice - i_comp - k_comp, e)
            removed += np.kron(i_comp + k_comp, e)
            
    return (SimpleChoiMatrix(ret), removed)

np.random.seed(10700)
s_star()