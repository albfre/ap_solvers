from ap_solvers import qp
from ap_solvers import dense_mp_matrix

from mpmath import mp
import scipy
import numpy as np
import time

import unittest
from parameterized import parameterized

class TestQP(unittest.TestCase):
  @parameterized.expand([dense_mp_matrix.matrix, mp.matrix])
  def test_unconstrained_qp(self, matrix):
    mp.dps = 100
    n = 10
    Q = matrix(n, n)
    c = matrix(n, 1)
    for i in range(n):
      c[i] = i
      for j in range(n):
        Q[i, j] = i * j
      Q[i, i] = n * n
    Q = Q + Q.T
    A_eq = []
    b_eq = []
    A_ineq = []
    b_ineq = []

    tol = mp.mpf('1e-50')
    x, f, res, gap, iteration = qp.solve_qp(Q, c, A_eq, b_eq, A_ineq, b_ineq, matrix, tol)
    self.assertTrue(abs(res) < tol)
    self.assertTrue(abs(gap) < tol)
    self.assertTrue(iteration == 1)

  @parameterized.expand([dense_mp_matrix.matrix, mp.matrix])
  def test_equality_constrained_qp(self, matrix):
    mp.dps = 100
    n = 10
    Q = matrix(n, n)
    c = matrix(n, 1)
    for i in range(n):
      c[i] = i
      for j in range(n):
        Q[i, j] = i * j
      Q[i, i] = n * n
    Q = Q + Q.T
    A_eq = matrix(1, n)
    b_eq = matrix(1, 1)
    A_eq[0, 0] = 1
    A_eq[0, 1] = 1
    A_ineq = []
    b_ineq = []

    tol = mp.mpf('1e-50')
    x, f, res, gap, iteration = qp.solve_qp(Q, c, A_eq, b_eq, A_ineq, b_ineq, matrix, tol)
    self.assertTrue(abs(res) < tol)
    self.assertTrue(abs(gap) < tol)
    self.assertTrue(iteration == 1)
    print('iteration %s' % iteration)

  def test_small_qp(self):
    mp.dps = 100
    n = 2
    t0 = 1300
    t1 = 50
    a00 = 809
    a01 = 359
    Q = mp.matrix([[a00 ** 2, a00 * a01], [a00 * a01, a01 **2]]) / t0 ** 2
    c = -mp.matrix([a00, a01]) / t0

    # e' x = 1
    A_eq = mp.ones(1, n)
    b_eq = mp.ones(1, 1)

    # x >= 0
    A_ineq = mp.diag([1, 1])
    b_ineq = mp.matrix(n, 1)

    tol = mp.mpf('1e-20')
    x, f, res, gap, iteration = qp.solve_qp(Q, c, A_eq, b_eq, A_ineq, b_ineq, mp.matrix, tol)
    self.assertTrue(abs(res) < tol)
    self.assertTrue(abs(gap) < tol)

  def test_larger_qp(self):
    mp.dps = 100
    n = 10
    Q = mp.matrix(n, n)
    c = mp.matrix(n, 1)
    A_ineq = mp.matrix(n, n)
    b_ineq = mp.matrix(n, 1)
    for i in range(n):
      Q[i, n-i-1] = 1.2 + 0.1
      Q[n-i-1, i] = 1.2 + 0.1

      Q[i, i] = i + 3
      c[i] = -0.5 * i
      A_ineq[i, i] = 1 # x[i] >= i
      b_ineq[i] = i * i * 0.01

    A_eq = mp.matrix(1, n)
    b_eq = mp.matrix(1, 1)
    A_eq[0, 1] = 1 # x[0] - 2 x[1] = 0
    A_eq[0, 2] = -2
    tol = mp.mpf('1e-20')

    tic = time.time()
    x, f, res, gap, iteration = qp.solve_qp(Q, c, A_eq, b_eq, A_ineq, b_ineq, mp.matrix, tol)
    toc = time.time() - tic
    self.assertTrue(abs(res) < tol)
    self.assertTrue(abs(gap) < tol)
    print('Time mp: ' + str(toc))

    tic = time.time()
    x2, f2, res2, gap2, iteration2 = qp.solve_qp(Q, c, A_eq, b_eq, A_ineq, b_ineq, dense_mp_matrix.matrix, tol)
    toc2 = time.time() - tic
    self.assertTrue(abs(res2) < tol)
    self.assertTrue(abs(gap2) < tol)

    print('Time dense: ' + str(toc2))
    print('Time factor: ' + str(toc/toc2))

    eps = mp.mpf('1e-' + str(int(mp.dps / 2)))
    self.assertTrue(abs(f - f2) < eps)
    self.assertTrue(abs(res - res2) < eps)
    self.assertTrue(abs(gap - gap2) < eps)
    self.assertEqual(iteration, iteration2)

if __name__ == "__main__":
  unittest.main()

