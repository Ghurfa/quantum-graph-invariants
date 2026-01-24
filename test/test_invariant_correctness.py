import numpy as np
import unittest

import graph_generate as gg
import invariant_interfaces as ii
import matrix_manip as mm
import subspace as ss

class LTCorrect(unittest.TestCase):
    def test_complete(self):
        for n in range(1, 7):
            actual, _ = ii.lt(gg.complete(n))
            self.assertAlmostEquals(actual, 1, delta=0.001)

    def test_independent(self):
        for n in range(1, 7):
            actual, _ = ii.lt(gg.independent(n))
            self.assertAlmostEquals(actual, n, delta=0.001)
    
    def test_line(self):
        for n in range(1, 7):
            actual, _ = ii.lt(gg.line(n))
            self.assertAlmostEquals(actual, 1 + (n - 1) // 2, delta=0.001)

    def test_c5(self):
        actual, _ = ii.lt(gg.cycle(5))
        self.assertAlmostEquals(actual, np.sqrt(5), delta=0.001)

    def test_random(self):
        from lovasz_stahlke import lovasz_theta

        np.random.seed(10700)
        for n in range(2, 8):
            for _ in range(n * n):
                G = gg.random(n)
                actual, _ = ii.lt(G)
                expected = lovasz_theta(G.adjacency_matrix)
                self.assertAlmostEquals(actual, expected, delta=0.001)

class QLTAndIndCPCorrect(unittest.TestCase):
    def correct_on_classical(self, invar):
        np.random.seed(10700)
        for n in range(2, 8):
            for _ in range(n):
                G = gg.random(n)
                expected, _ = ii.lt(G)
                actual, _ = invar(ss.sg(G))
                self.assertAlmostEquals(actual, expected, delta = 0.001)

    def invariant_wrt_conjugation(self, invar):
        np.random.seed(10700)

        for n in range(3, 8):
            for _ in range(5):
                myUni = np.matrix(mm.rand_uni(n))
                myUniH = myUni.getH()

                quantum_graph, _, _ = ss.random_s1_s2(n)

                newbasis = []
                for element in quantum_graph.basis:
                    prod1 = np.dot(myUni, element)
                    prod2 = np.dot(prod1, myUniH)
                    newbasis.append(np.array(prod2))

                conjugatedSubspace = ss.from_basis(newbasis)

                expected, _ = invar(quantum_graph)
                actual, _ = invar(conjugatedSubspace)

                self.assertAlmostEquals(actual, expected, delta = 0.002)

    def test_qlt_correct_on_classical(self):
        self.correct_on_classical(ii.qlt)

    def test_ind_cp_correct_on_classical(self):
        self.correct_on_classical(lambda x: ii.ind_cp(ss.mn(x.n), x))

    def test_qlt_invariant_wrt_conjugation(self):
        self.invariant_wrt_conjugation(ii.qlt)

    def test_ind_cp_invariant_wrt_conjugation(self):
        self.invariant_wrt_conjugation(lambda x: ii.ind_cp(ss.mn(x.n), x))

    def test_distinguish_qlt_from_ind_cp(self):
        for n in range(2, 6):
            qg = ss.from_constraints([np.ones((n, n)) - np.eye(n)])
            actual_qlt, _ = ii.qlt(qg)
            actual_ind_cp, _ = ii.ind_cp(ss.mn(n), qg)
            self.assertAlmostEquals(actual_qlt, n, delta = 0.001)
            self.assertAlmostEquals(actual_ind_cp, 2, delta = 0.001)
