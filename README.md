# ap_solvers

Arbitrary precision solvers using mpmath. It provides the following solvers:

## Bunch-Kaufman solver

The Bunch-Kaufman solver is a matrix factorization solver for symmetric indefinite matrices.

Usage:
```python
# Example code for using the quadratic programming solver

from ap_solvers import bunch_kaufman
from mpmath import mp

mp.dps = 50 # Set decimal precision

A = mp.matrix([[4, 2, -2], [2, 5, 6], [-2, 6, 5]])
b = mp.matrix([1, 2, 3])

L, ipiv, info = bunch_kaufman.overwriting_symmetric_indefinite_factorization(A)
x = bunch_kaufman.overwriting_solve_using_factorization(L, ipiv, b)

print(str(x))
```

## Quadratic programming solver

The primal dual interior point solver can solve optimization problems with convex quadratic objective functions and linear equality and inequality constraints.

Usage:
```python
# Example code for using the quadratic programming solver

from ap_solvers import qp, dense_mp_matrix
from mpmath import mp

mp.dps = 50 # Set decimal precision

Q = dense_mp_matrix.matrix([[809 ** 2, 809 * 359], [809 * 359, 359 **2]]) / 1300 ** 2
c = -dense_mp_matrix.matrix([809, 359]) / 1300

# e' x = 1
A_eq = mp.ones(1, n)
b_eq = mp.ones(1, 1)

# x >= 0
A_ineq = mp.diag([1, 1])
b_ineq = mp.matrix(n, 1)

x, f, res, gap, iteration = qp.solve_qp(Q, c, A_eq, b_eq, A_ineq, b_ineq, dense_mp_matrix)

print(str(f))
```
