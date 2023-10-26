from src import qp

import unittest
import scipy
import numpy as np
from mpmath import mp

class TestQP(unittest.TestCase):
  def test_qp(self):
    n = 2
    Q = mp.matrix(n, n)
    c = mp.matrix(n, 1)
    t0 = 1300;
    t1 = 50;
    a00 = 809;
    a01 = 359;
    a10 = 25;
    a11 = 77;

    # e' x = 1
    A_eq = mp.ones(1, n)
    b_eq = mp.ones(1, 1)

    # x >= 0
    A_ineq = mp.matrix(n, n)
    for i in range(n):
      A_ineq[i, i ] = mp.mpf('1')

    b_ineq = mp.matrix(n, 1)

    k = 0;
    Q[0, 0] = (a00**2) / t0**2 + k;
    Q[1, 1] = (a01**2) / t0**2 + k;
    Q[0, 1] = (a00 * a01) / t0**2;
    Q[1, 0] = (a00 * a01) / t0**2;
    c[0] = -t0 * a00 / t0**2;
    c[1] = -t0 * a01 / t0**2;
    print('Q')
    print(Q)

    qp.solve_qp(Q, c, A_eq, b_eq, A_ineq, b_ineq);


if __name__ == "__main__":
  unittest.main()

