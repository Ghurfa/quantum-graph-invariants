import numpy as np
import matrix_manip as mm
import subspace as ss
import matplotlib.pyplot as plt
import networkx as nx
import invariants as invar


def visualize_graph(adjMat: np.ndarray):
    #Visualize graph function.
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
    for n in range(3,8):
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
            
