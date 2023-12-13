from ap_solvers import sqp
from ap_solvers import dense_mp_matrix

from mpmath import mp
import scipy
import numpy as np
import time
import random

import unittest
from parameterized import parameterized

class TestSQP(unittest.TestCase):
  @parameterized.expand([dense_mp_matrix.matrix, mp.matrix])
  def ap_test_sqp(self, matrix):
    mp.dps = 100

    f = lambda x: (x[0] - 1)**2 + (x[1] - 0.25)**2 # (x1-1)^2 + (x2-0.25)^2
    c = lambda x: -(x[0] ** 2 + x[1] ** 2 - mp.one) # x1^2 + x2^2 <= 1

    tol = mp.mpf('1e-20')
    opt = sqp.Sqp(f, None, [c], None, tol=tol, matrix=matrix, print_stats=True)
    x0 = [1, 1]
    tic = time.time()
    x, f, cs, status = opt.solve(x0, 100)
    toc = time.time() - tic
    print_dps = 10
    print(status)
    f_str = mp.nstr(f, print_dps)
    print('f: %s' % f_str)
    for i in range(len(cs)):
      cs_str = mp.nstr(cs[i], print_dps)
      print('c[%s]: %s' % (i, cs_str))

    for i in range(len(x)):
      x_str = mp.nstr(x[i], print_dps)
      print('x[%s]: %s' % (i, x_str))
    print('Time to solve: %s' % toc)

  @parameterized.expand([dense_mp_matrix.matrix])
  def test_larger_sqp(self, matrix):
    mp.dps = 100
    random.seed(17)
    n_vox = 50
    n_bix = 5
    target = range(10, 20)
    oar = range(5, 15)
    dose_matrix = matrix(n_vox, n_bix)
    dose_matrix_T = dose_matrix.T

    for i in range(n_vox):
      for j in range(n_bix):
        dose_matrix[i, j] = random.random()

    def f(x):
      val = mp.zero
      dose = dose_matrix * x
      for i in target:
        val += 100 * (dose[i] - mp.one)**2
      for i in oar:
        val += 10 * (dose[i] - mp.zero)**2
      for i in range(n_vox):
        val += 1 * (dose[i] - mp.zero)**2
      return val

    def grad_f(x):
      grad_wrt_dose = matrix([mp.zero] * n_vox)
      dose = dose_matrix * x
      for i in target:
        grad_wrt_dose[i] += 2 * 100 * (dose[i] - mp.one)
      for i in oar:
        grad_wrt_dose[i] += 2 * 10 * (dose[i] - mp.zero)
      for i in range(n_vox):
        grad_wrt_dose[i] += 2 * 1 * (dose[i] - mp.zero)
      return dose_matrix_T * grad_wrt_dose


    def c(x):
      val = mp.zero
      dose = dose_matrix * x
      for i in target:
        val += min(dose[i] - 0.8 * mp.one, mp.zero)**2
        #val += (dose[i])**2
      return -val + mp.mpf('1e-3') # minus sing to get min(d - 0.8, 0)^2 <= 0
        
    tol = mp.mpf('1e-20')
    opt = sqp.Sqp(f, None, [c], None, tol=tol, matrix=matrix, print_stats=True)
    x0 = [1] * n_bix
    tic = time.time()
    x, f, cs, status = opt.solve(x0, 400)
    toc = time.time() - tic
    print_dps = 10
    print(status)
    f_str = mp.nstr(f, print_dps)
    print('f: %s' % f_str)
    for i in range(len(cs)):
      cs_str = mp.nstr(cs[i], print_dps)
      print('c[%s]: %s' % (i, cs_str))

    for i in range(len(x)):
      x_str = mp.nstr(x[i], print_dps)
      print('x[%s]: %s' % (i, x_str))
    print('Time to solve: %s' % toc)

if __name__ == "__main__":
  unittest.main()

