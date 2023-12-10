from ap_solvers import sqp
from ap_solvers import dense_mp_matrix

from mpmath import mp
import scipy
import numpy as np
import time

import unittest
from parameterized import parameterized

class TestSQP(unittest.TestCase):
  @parameterized.expand([dense_mp_matrix.matrix])
  def test_sqp(self, matrix):
    mp.dps = 100

    f = lambda x: (x[0] - 1)**2 + (x[1] - 0.25)**2
    c = lambda x: x[0] ** 2 + x[1] ** 2 - mp.one

    tol = mp.mpf('1e-50')
    opt = sqp.Sqp(f, None, [c], None)
    x0 = [2, 2]
    x, f, cs = opt.solve(x0)
    print('f: %s' % f)
    print('cs: %s' % cs)
    print('x: %s' % x)

if __name__ == "__main__":
  unittest.main()

