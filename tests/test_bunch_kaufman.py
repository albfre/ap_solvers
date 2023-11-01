from ap_solvers import bunch_kaufman
from ap_solvers import dense_mp_matrix

from mpmath import mp
import scipy
import numpy as np
import time

import unittest
from parameterized import parameterized

class TestBunchKaufman(unittest.TestCase):
  def setup_factorization_test(self):
    mp.dps = 100
    A = [[4, 2, -2],
         [2, 5, 6],
         [-2, 6, 5]]
    b = [1, 2, 3]
    """
        A = [[4,  2, -2, 2, 2, 2, 2, 2, 2],
             [2,  5,  6, 6, 6, 6, 6, 6, 6],
             [-2, 6,  5, 5, 5, 5, 5, 5, 5],
             [2,  6,  5, 2, 3, 3, 3, 3, 3],
             [2,  6,  5, 3, 2, 4, 4, 4, 4],
             [2,  6,  5, 3, 4, 8, 8, 8, 8],
             [2,  6,  5, 3, 4, 8, 9, 9, 9],
             [2,  6,  5, 3, 4, 8, 9, 1, 1],
             [2,  6,  5, 3, 4, 8, 9, 1, 2]]

        b = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    """

    # Compute expected result using scipy
    A_np = np.array(A)
    b_np = np.array(b)

    L_sp, D_sp, perm_sp = scipy.linalg.ldl(A_np)
    lu, piv = scipy.linalg.lu_factor(A_np)
    x_sp = scipy.linalg.lu_solve((lu, piv), b_np)

    return A, b, x_sp

  def assert_solution(self, x_sp, x_mp):
    one_over_twelve = 1 / mp.mpf("12")
    eps = mp.mpf("1e-30")

    self.assertTrue(abs(x_mp[0] + one_over_twelve) < eps, "Expected x[0] to equal 1/12 to high precision")

    for i in range(len(x_mp)):
      self.assertTrue(abs(x_mp[i] - x_sp[i]) < 1e-15, "Expected %s to equal %s for i=%s" % (x_mp[i], x_sp[i], i))

  @parameterized.expand([False, True])
  def test_bunch_kaufman(self, use_mp):
    print("Running test with use_mp=%s" % use_mp)
    A, b, x_sp = self.setup_factorization_test()
    matrix = mp.matrix if use_mp else dense_mp_matrix.matrix

    # Compute result using bunch_kaufman implementation
    tic = time.time()
    L, ipiv, info = bunch_kaufman.overwriting_symmetric_indefinite_factorization(matrix(A))
    x_mp = bunch_kaufman.overwriting_solve_using_factorization(L, ipiv, matrix(b))
    toc = time.time() - tic
    print("Time: " + str(toc))
    
    self.assert_solution(x_sp, x_mp)

if __name__ == "__main__":
  unittest.main()

