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

    f = lambda x: (x[0] - 1)**2 + (x[1] - 0.25)**2 # (x1-1)^2 + (x2-0.25)^2
    c = lambda x: -(x[0] ** 2 + x[1] ** 2 - mp.one) # x1^2 + x2^2 <= 1

    tol = mp.mpf('1e-50')
    opt = sqp.Sqp(f, None, [c], None)
    x0 = [1, 1]
    x, f, cs = opt.solve(x0)
    print_dps = 10
    f_str = mp.nstr(f, print_dps)
    print('f: %s' % f_str)
    for i in range(len(cs)):
      cs_str = mp.nstr(cs[i], print_dps)
      print('c[%s]: %s' % (i, cs_str))

    for i in range(len(x)):
      x_str = mp.nstr(x[i], print_dps)
      print('x[%s]: %s' % (i, x_str))

if __name__ == "__main__":
  unittest.main()

