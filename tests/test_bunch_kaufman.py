from src import bunch_kaufman

import unittest
import scipy
import numpy as np
from mpmath import mp

class TestBunchKaufman(unittest.TestCase):
  def test_bunch_kaufman(self):
    A = [[4, 2, -2],
         [2, 5, 6],
         [-2, 6, 5]]
    b = [1, 2, 3]

    # Create expected result using scipy
    A_np = np.array(A)
    b_np = np.array(b)

    L_sp, D_sp, perm_sp = scipy.linalg.ldl(A_np)
    lu, piv = scipy.linalg.lu_factor(A_np)
    x_sp = scipy.linalg.lu_solve((lu, piv), b_np)

    # Compute result using bunch_kaufman implementation
    mp.dps = 100
    A_mp = mp.matrix(A)
    b_mp = mp.matrix(b)

    L, ipiv, info = bunch_kaufman.symmetric_indefinite_factorization(A_mp)
    x_mp = bunch_kaufman.solve_using_factorization(L, ipiv, b_mp)

    one_over_twelve = 1 / mp.mpf("12")
    eps = mp.mpf("1e-30")
    mp.nprint(x_mp, mp.dps)

    self.assertTrue(abs(x_mp[0] + one_over_twelve) < eps, "Expected x[0] to equal 1/12 to high precision")

    for i in range(len(b)):
      self.assertTrue(abs(x_mp[i] - x_sp[i]) < 1e-15, "Expected %s to equal %s for i=%s" % (x_mp[i], x_sp[i], i))


if __name__ == "__main__":
  unittest.main()

