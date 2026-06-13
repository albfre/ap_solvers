# ap_solvers

Arbitrary precision solvers using mpmath. It provides the following solvers:

* Sequential quadratic programming (SQP) solver
* Quadratic programming (QP) solver
* Nelder-Mead solver
* Convex hull solver
* Bunch-Kaufman solver

## SQP solver

The SQP solver can solve constrained nonlinear optimization problems on the form
$$\underset{x}{\text{minimize }} f(x) \text{ subject to } g_i(x) \geq 0 \text{ for all } i.$$ If gradients are not provided they are approximated by finite differences.

Usage:
```python
# Example code for using the sequential quadratic programming solver

from ap_solvers import sqp, dense_mp_matrix
from mpmath import mp

mp.dps = 50 # Set decimal precision

f = lambda x: (x[0] - 1)**2 + (x[1] - 0.25)**2 # (x1-1)^2 + (x2-0.25)^2
c = lambda x: -(x[0] ** 2 + x[1] ** 2 - mp.one) # x1^2 + x2^2 <= 1

x0 = dense_mp_matrix.matrix([mp.mpf('0.5'), mp.mpf('0.5')])
tol = mp.mpf('1e-20')
opt = sqp.Sqp(f, None, [c], None, tol=tol, matrix=dense_mp_matrix.matrix, print_stats=True)
x, f_val, cs, status = opt.solve(x0, max_iter=100)

print(str(f_val))
```

## QP solver

The primal dual interior point solver can solve optimization problems with convex quadratic objective functions and linear equality and inequality constraints. It solves problems on the form

$$\underset{x}{\text{minimize }} 0.5 x' H x + c' x \text{ subject to } A_{\text{eq}} x = b_{\text{eq}} \text{ and } A_{\text{ineq}} x \geq b_{\text{ineq}}.$$

Usage:
```python
# Example code for using the quadratic programming solver

from ap_solvers import qp, dense_mp_matrix
from mpmath import mp

mp.dps = 50 # Set decimal precision

n = 2
Q = dense_mp_matrix.matrix([[809 ** 2, 809 * 359], [809 * 359, 359 **2]]) / 1300 ** 2
c = -dense_mp_matrix.matrix([809, 359]) / 1300

# e' x = 1
A_eq = mp.ones(1, n)
b_eq = mp.ones(1, 1)

# x >= 0
A_ineq = mp.diag([1, 1])
b_ineq = mp.matrix(n, 1)

result = qp.solve_qp(Q, c, A_eq, b_eq, A_ineq, b_ineq, dense_mp_matrix.matrix)

print(str(result.objective))
```

## Nelder-Mead solver

The Nelder-Mead solver is a derivative-free optimization algorithm that can minimize unconstrained scalar functions. It uses a simplex-based approach and does not require gradient information, making it suitable for non-smooth or noisy functions.

Usage:
```python
# Example code for using the Nelder-Mead solver

from ap_solvers import nelder_mead
from mpmath import mp

mp.dps = 50 # Set decimal precision

# Define the objective function (e.g., Rosenbrock function)
f = lambda x: (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

# Starting point
x_start = [mp.mpf('0.0'), mp.mpf('0.0')]

# Optimize with verbose output
x, f_val, status = nelder_mead.nelder_mead(
    f, x_start, 
    max_iter=5000, 
    no_improve_thr=mp.mpf('1e-40'),
    verbose=True
)

print(f"Optimal point: {x}")
print(f"Optimal value: {f_val}")
print(f"Status: {status}")
```

## Convex hull solver

The convex hull solver is based on the quickhull algorithm. It can compute convex hulls in arbitrary dimensions.

Usage:
```python
# Example code for using the convex hull solver

from ap_solvers import convex_hull
from mpmath import mp

mp.dps = 50 # Set decimal precision

points = [[0, 0, 0],
          [0, 0, 1], 
          [0, 1, 0], 
          [1, 0, 0], 
          [0.2, 0.2, 0.2], 
          [0.7, 0.7, 0.7], 
          [0.3, 0.3, 0.3]]

# Compute the convex hull
ch = convex_hull.ConvexHull(points)
print(ch.simplices)
print(ch.equations)
```

## Bunch-Kaufman solver

The Bunch-Kaufman solver is a matrix factorization solver for symmetric indefinite matrices.

Usage:
```python
# Example code for using the Bunch-Kaufman solver

from ap_solvers import bunch_kaufman
from mpmath import mp

mp.dps = 50 # Set decimal precision

A = mp.matrix([[4, 2, -2], [2, 5, 6], [-2, 6, 5]])
b = mp.matrix([1, 2, 3])

L, ipiv, info = bunch_kaufman.overwriting_symmetric_indefinite_factorization(A)
x = bunch_kaufman.overwriting_solve_using_factorization(L, ipiv, b)

print(str(x))
```

