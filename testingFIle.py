import numpy as np
import matrix_manip as mm
import subspace as ss
import invariant_interfaces as invar
import graph_generate as gg

def visualize_graph(adjMat: np.ndarray):
    """
    Visualize graph function.
    """

    import matplotlib.pyplot as plt
    import networkx as nx

    def gen_graph(adjacencyMat):
        rows, cols = np.where(adjacencyMat==1)
        edges = zip(rows.tolist(), cols.tolist())
        gr = nx.Graph()
        gr.add_edges_from(edges, labels=True)
        return gr
    
    def show_graph(graph):
        nx.draw(graph, node_size=500)
        plt.show()

    show_graph(gen_graph(adjMat))

def check_conjugation():
    """
    Verifies that our ind_cp implementation is invariant under the input being conjugated by a unitary matrix
    """

    for n in range(3, 8):
        universe = ss.mn(n)
        print("Size is now", n, "\n")
        for i in range(5):
            myUni = np.matrix(mm.rand_uni(n))
            myUniH = myUni.getH()

            quantum_graph, _, _ = ss.random_s1_s2(n)

            newbasis = []
            for element in quantum_graph.basis:
                prod1 = np.dot(myUni, element)
                prod2 = np.dot(prod1, myUniH)
                newbasis.append(np.array(prod2))

            conjugatedSubspace = ss.from_basis(newbasis)

            print(abs(invar.ind_cp(universe, quantum_graph)[0]-invar.ind_cp(universe, conjugatedSubspace)[0]), "\n")


def get_conjugate_basis(myUni, basis):
    """
    Given compatible unitary matrix and a basis of matrices, say {b_i}, define a basis to be {Ub_iU*} 
    """
    myUniH = myUni.getH()
    newbasis = []
    for element in basis:
        prod1 = np.dot(myUni, element)
        prod2 = np.dot(prod1, myUniH)
        newbasis.append(np.array(prod2))
    return newbasis


def sg1_rotate_sg2(unitary, sg1, sg2):
    """
    Given two graph systems and a compatible unitary matrix, define the quantum graph given by Usg1U* + Sg2
    #Jordan L checked this function implementation to his original test implementation.
    """
    myQuantumBasis = get_conjugate_basis(unitary, sg1.basis)
    ss.extend_basis(myQuantumBasis, iter(sg2.basis))
    myQuantumGraph = ss.from_basis(myQuantumBasis)
    return myQuantumGraph
    

n = 5
myUni = np.matrix(mm.rand_uni(n))
myUniH = myUni.getH()

for i in range(10):
    graph1 = gg.random(n, 0.2)
    graph2 = gg.random(n, 0.2)


    #visualize_graph(graph1.adjacency_matrix[0])
    #visualize_graph(graph2.adjacency_matrix[0])

    graphsystem1 = ss.sg(graph1)
    graphsystem2 = ss.sg(graph2)

    print(ss.is_independent(np.identity(n), graphsystem2.basis))

    myQgraph = sg1_rotate_sg2(myUni, graphsystem1, graphsystem2)



    print("\nquantum theta of composite", invar.qlt(myQgraph)[0])
    print("\nLovasz theta of graph 1", invar.lt_general(graphsystem1)[0])
    print("\nLovasz theta of graph 2", invar.lt_general(graphsystem2)[0])
