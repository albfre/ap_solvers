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
  def _solve_and_print(self, opt, x0, max_iter):
    tic = time.time()
    x, f, cs, status = opt.solve(x0, max_iter)
    toc = time.time() - tic

    #self.assertTrue(status == "Optimal solution found")

    print_dps = 10
    print(status)
    f_str = mp.nstr(f, print_dps)
    print('f: %s' % f_str)
    opt._compute_gradients(x)
    f_grad_str = [mp.nstr(fi, print_dps) for fi in opt._f_grad_k]
    print('fgrad: %s' % f_grad_str)

    for i in range(len(x)):
      x_str = mp.nstr(x[i], print_dps)
      print('x[%s]: %s' % (i, x_str))
    print('Time to solve: %s' % toc)

  @parameterized.expand([dense_mp_matrix.matrix, mp.matrix])
  def test_sqp(self, matrix):
    mp.dps = 100

    f = lambda x: (x[0] - 1)**2 + (x[1] - 0.25)**2 # (x1-1)^2 + (x2-0.25)^2
    c = lambda x: -(x[0] ** 2 + x[1] ** 2 - mp.one) # x1^2 + x2^2 <= 1
    tol = mp.mpf('1e-20')
    opt = sqp.Sqp(f, None, [c], None, tol=tol, matrix=matrix, print_stats=True)
    x0 = [1, 1]
    self._solve_and_print(opt, x0, 100)

  @parameterized.expand([mp.matrix])
  def test_unconstrained_sqp(self, matrix):
    mp.dps = 100

    n = 6
    f = lambda x: abs(x[0] - 1)**n + abs(x[1] - 0.25)**n # (x1-1)^6 + (x2-0.25)^4
    tol = mp.mpf('1e-30')
    opt = sqp.Sqp(f, None, [], None, tol=tol, hessian_reset_iter=50, matrix=matrix, print_stats=True)
    x0 = [10, 10]
    self._solve_and_print(opt, x0, 1000)

  @parameterized.expand([dense_mp_matrix.matrix])
  def test_larger_sqp(self, matrix):
    mp.dps = 100
    random.seed(17)
    n_vox = 50
    n_bix = 5
    target = range(10, 20)
    oar = range(5, 15)
    dose_matrix = matrix(n_vox, n_bix)

    for i in range(n_vox):
      for j in range(n_bix):
        dose_matrix[i, j] = random.random()
    dose_matrix_T = dose_matrix.T

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

    def f_grad(x):
      grad_wrt_dose = matrix(n_vox, 1)
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
      return -val

    def c_grad(x):
      grad_wrt_dose = matrix(n_vox, 1)
      dose = dose_matrix * x
      for i in target:
        grad_wrt_dose[i] += 2 * min(dose[i] - 0.8 * mp.one, 0)
      return -dose_matrix_T * grad_wrt_dose
        
    tol = mp.mpf('1e-2')
    opt = sqp.Sqp(f, f_grad, [c], [c_grad], tol=tol, matrix=matrix, print_stats=True)
    x0 = matrix([1] * n_bix)
    self._solve_and_print(opt, x0, 100)

  def _test_scipy_sqp(self):
    mp.dps = 100
    random.seed(17)
    n_vox = 50
    n_bix = 5
    target = range(10, 20)
    oar = range(5, 15)
    dose_matrix = np.array([[0.0]*n_bix for i in range(n_vox)])

    for i in range(n_vox):
      for j in range(n_bix):
        dose_matrix[i, j] = random.random()
    dose_matrix_T = dose_matrix.T

    def f(x):
      val = 0
      dose = dose_matrix @ x
      for i in target:
        val += 100 * (dose[i] - 1)**2
      for i in oar:
        val += 10 * (dose[i] - 0)**2
      for i in range(n_vox):
        val += 1 * (dose[i] - 0)**2
      return val

    def f_grad(x):
      grad_wrt_dose = matrix(n_vox, 1)
      dose = dose_matrix * x
      for i in target:
        grad_wrt_dose[i] += 2 * 100 * (dose[i] - mp.one)
      for i in oar:
        grad_wrt_dose[i] += 2 * 10 * (dose[i] - mp.zero)
      for i in range(n_vox):
        grad_wrt_dose[i] += 2 * 1 * (dose[i] - mp.zero)
      return dose_matrix_T * grad_wrt_dose

    def c(x):
      val = 0
      dose = dose_matrix @ x
      for i in target:
        val += min(dose[i] - 0.8 * 1, 0)**2
      return -val #-val + mp.mpf('1e-6') # minus sing to get min(d - 0.8, 0)^2 <= 0

    def c_grad(x):
      grad_wrt_dose = matrix(n_vox, 1)
      dose = dose_matrix * x
      for i in target:
        grad_wrt_dose[i] += 2 * min(dose[i] - 0.8 * mp.one, 0)
      return -dose_matrix_T * grad_wrt_dose

    constraints = ({'type': 'ineq', 'fun': c})

# Optimization using SLSQP
        
    x0 = np.array([1.0] * n_bix)
    print(str(x0))
    tic = time.time()
    options = {'maxiter':5000}
    result = scipy.optimize.minimize(f, x0, method='SLSQP', constraints=constraints, tol=1e-25, options=options)
    toc = time.time() - tic
    print_dps = 10
    print(result)
    print(str(c(result.x)))
    print('Time to solve: %s' % toc)

if __name__ == "__main__":
  unittest.main()

