# Parts of this file were copy-pasted from graph_generate.py by Colton Griffin and Aneesh Vinod Khilnani (https://github.com/chemfinal-dot/2021-Sinclair-Project)

import operator
import numpy as np
from typing import *
from graph_generate import Graph

def e_matrix(n: int, i: int, j: int) -> np.ndarray:
    """
    Creates a matrix that is all zeroes except for a one at (i, j)
    """

    E = np.zeros((n,n), dtype=int)
    E[i, j] = 1
    return E

def delta_matrix(n: int) -> np.ndarray:
    """
    Creates a matrix that is the sum of (e (x) e) where e is each of the e-matrices of size n
    """

    ret = np.zeros((n * n, n * n), dtype=int)
    for i in range(n):
        for j in range(n):
            ret[i * n + i, j * n + j] = 1
    return ret

def rand_uni(n) -> np.ndarray:
    """
    Generates an n x n random matrix with Gaussian-distributed entries (clipped to [0,1])
    and then orthonormalizes its columns using the Gram-Schmidt process to form a unitary matrix.
    """
    X = np.random.normal(loc=0, scale=1, size=(n, n))
    
    Q = np.zeros_like(X)
    for i in range(X.shape[1]):
        q = X[:, i].copy() 
        for j in range(i):
            q -= np.dot(Q[:, j], X[:, i]) * Q[:, j]
        q /= np.linalg.norm(q)
        Q[:, i] = q
        
    return Q

def _delegate_op(numpy_op):
    """Decorator to apply a numpy operation to self.data and return an instance of self.__class__."""
    def wrapper(self, other):
        other_data = other.data if isinstance(other, SimpleMatrix) else other
        result = numpy_op(self.data, other_data)
        return self.__class__(result)
    return wrapper

class SimpleMatrix:
    def __init__(self, mat, precision: int = 2):
        self.data = np.array(mat)
        self.precision = precision
    
    def _entry_str(self, i, j):
        col_width = 3 + self.precision
        val = round(self.data[i, j], self.precision)
        ret = f"{val:.{self.precision}f}".rjust(col_width)
        if val == 0:
            return "  ." + " " * (col_width - 3)
        if ret[-self.precision:] == "0" * self.precision:
            ret = ret[:-self.precision] + " " * self.precision
        return ret
    
    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getattr__(self, attr):
        """Catch undefined attributes and delegate to self.data"""

        result = getattr(self.data, attr)
        return self.__class__(result) if isinstance(result, np.ndarray) else result

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Enable combining SimpleMatrix with numpy arrays in expressions like adding
        """
        inputs = tuple(i.data if isinstance(i, SimpleMatrix) else i for i in inputs)
        result = getattr(ufunc, method)(*inputs, **kwargs)
        return self.__class__(result) if isinstance(result, np.ndarray) else result
    
    __add__ = _delegate_op(operator.add)
    __radd__ = _delegate_op(operator.add)
    __sub__ = _delegate_op(operator.sub)
    __rsub__ = _delegate_op(operator.sub)
    __mul__ = _delegate_op(operator.mul)
    __rmul__ = _delegate_op(operator.mul)
    __truediv__ = _delegate_op(operator.truediv)
    __rtruediv__ = _delegate_op(operator.truediv)

    def __str__(self):
        height, width = self.data.shape
        def row(i):
            return " ".join(self._entry_str(i, j) for j in range(0, width))
        return "\n".join(row(i) for i in range(0, height))
    
    def __repr__(self):
        return self.__str__()
    
    @property
    def eigs(self):
        return SimpleMatrix(np.linalg.eigvals(self.data))
    
class SimpleSymmMatrix(SimpleMatrix):
    def __init__(self, mat: np.ndarray, precision: int = 2):
        SimpleMatrix.__init__(self, mat, precision)

    @property
    def eigs(self):
        return SimpleMatrix(np.linalg.eigvalsh(self.data))

class SimpleChoiMatrix(SimpleSymmMatrix):
    def __init__(self, mat: np.ndarray, precision: int = 2):
        SimpleSymmMatrix.__init__(self, mat, precision)
        self.n = int(mat.shape[0] ** 0.5)
    
    def __str__(self):
        col_width = 3 + self.precision
        def col_sect(i, j):
            return " ".join(self._entry_str(i, j * self.n + k) for k in range(0, self.n))
        def row(i):
            return " | ".join(col_sect(i, j) for j in range(0, self.n))
        def row_sect(i):
            return "\n".join(row(i * self.n + j) for j in range(0, self.n))
        row_sects = [row_sect(i) for i in range(0, self.n)]
        return ("\n" + "-" * ((col_width + 1) * (self.n * self.n) + (3 - 1) * self.n - 3) + "\n").join(row_sects)

    
def compress(matrix: SimpleChoiMatrix):
    """
    Obtains the matrix A, defined as A_ij = matrix_{i * n + i, j * n + j}
    """

    if not(isinstance(matrix, SimpleChoiMatrix)):
        raise ValueError("compress() expects a Choi (n^2 by n^2) matrix")
    
    n = matrix.n
    return SimpleMatrix(np.array([[matrix[i * n + i, j * n + j] for j in range(0, n)] for i in range(0, n)]))
