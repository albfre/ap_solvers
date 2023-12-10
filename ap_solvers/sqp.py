from ap_solvers import dense_mp_matrix, qp
from mpmath import mp

class Sqp:
  def __init__(self, f, f_grad, cs, c_grads, tol = mp.mpf('1e-20'), matrix = dense_mp_matrix.matrix):
    """
    Initialize the Sqp object with a set of points.

    minimize f(x)
    s.t. c_i(x) <= 0 for all i

    if f_grad or c_grad is None, they will be approximated by finite differences
    """
    self.f = f
    self.f_grad = f_grad
    self.cs = cs if cs else []
    self.c_grads = c_grads if c_grads else []
    self.rho = [mp.zero] * len(self.cs)
    self.tol = tol
    self.minor_tol = tol
    self.matrix = matrix
    self.print_stats = True
    self.eta = mp.mpf('0.5')
    self.x_used_for_gradient_computation = None

  def solve(self, x0, max_iter = 20):
    n = len(x0)
    self.hessian_approximation = self.matrix(n, n)
    for i in range(n):
      self.hessian_approximation[i, i] = mp.one

    x_k = self.matrix(x0)
    s_k = self.matrix(len(self.cs), 1)
    pi_k = self.matrix(len(self.cs), 1)

    for iteration in range(max_iter):
      self._compute_gradients(x_k)
      print('solve qp for iteration %s' % iteration)
      x_hat, s_hat, pi_hat, minor_iterations = self._solve_qp(x_k)
      x_next, s_next, pi_next, alpha = self._line_search(x_k, s_k, pi_k, x_hat, s_hat, pi_hat)
      self._update_hessian_approximation(x_k, x_next, x_hat, pi_next, alpha)
      x_k, s_k, pi_k = x_next, s_next, pi_next

      self._print(iteration, x_k, alpha, minor_iterations)

      if self._check_convergence(x_k, pi_k):
        if self.print_stats:
          print('Optimal solution found')
        break
    return x_k, self.f(x_k), self.evaluate_constraints(x_k)

  def evaluate_constraints(self, x):
    return self.matrix([c(x) for c in self.cs])

  def _print(self, iteration, x, alpha, minor_iterations):
    print_dps = 10
    obj = mp.nstr(self.f(x), print_dps)
    step = mp.nstr(alpha, print_dps)
    c = mp.nstr(max(self.evaluate_constraints(x)), print_dps)
    print('iter=%s, step=%s minors=%s, obj=%s, constr=%s' % (iteration, step, minor_iterations, obj, c))

  def _check_convergence(self, x, pi):
    tau_x = self.tol * (1 + max(x))
    tau_pi = self.tol * (1 + max(pi))
    for i in range(len(self.cs)):
      if pi[i] < -tau_pi: return False

      ci = self.cs[i](x)
      if ci > tau_x: return False
      if ci * pi[i] > tau_pi: return False

      self._compute_gradients(x)
      d = self.f_grad_k + self.jacobian_k.T * pi
      if any(abs(di) > tau_pi for di in d): return False
    return True


  def _update_hessian_approximation(self, x0, x1, x_hat, pi1, alpha):
    f_grad0 = self.f_grad_k.copy()
    jacobian0 = self.jacobian_k.copy()

    self._compute_gradients(x1)
    delta = x1 - x0
    y = self.f_grad_k - f_grad0 + (self.jacobian_k - jacobian0).T * pi1

    p = x_hat - x0 # search direction
    sigma = alpha * (mp.one - self.eta) * p.T * self.hessian_approximation * p

    yTdelta = (y.T * delta)[0]

    if yTdelta >= sigma[0]:
      q = self.hessian_approximation * delta
      qTdelta = (q.T * delta)[0]
      self.hessian_approximation += y * y.T / yTdelta + q * q.T / qTdelta
    elif self.print_stats:
      print("No hessian update. y' delta = %s, sigma = %s" % (yTdelta, sigma))


  def _finite_difference_grad(self, f, x):
    h = self.tol
    grad = self.matrix(len(x), 1)
    x_shift = x.copy()
    for i in range(len(x)):
      x_shift[i] = x[i] + h
      fp = f(x_shift)
      x_shift[i] = x[i] - h
      fm = f(x_shift)
      x_shift[i] = x[i]
      grad[i] = (fp - fm) / (2 * h)
    return grad

  def _compute_gradients(self, x):
    if x == self.x_used_for_gradient_computation:
      return

    self.f_grad_k = self.f_grad(x) if self.f_grad else self._finite_difference_grad(self.f, x)
    self.jacobian_k = self.matrix(len(self.cs), len(x))
    for i in range(len(self.cs)):
      if self.c_grads and self.c_grads[i]:
        grad = self.c_grads[i](x)
      else:
        grad = self._finite_difference_grad(self.cs[i], x)
      for j in range(len(grad)):
        self.jacobian_k[i, j] = grad[j]
    self.x_used_for_gradient_computation = x
    
  def _solve_qp(self, x_k):
    """ minimize f_k + g_k'(x - x_k) + 0.5 (x - x_k)' H_k (x - x_k)
        s.t. c_k + J_k (x - x_k) <= 0

        i.e., 

        minimize (g_k - H_k x_k)' x + 0.5 x' H_k x + constants
        s.t. J_k x <= J_k x_k - c_k
    """
    Q = self.hessian_approximation
    
    c = self.f_grad_k - Q * x_k
    A_eq = []
    b_eq = []
    self._compute_gradients(x_k)
    A_ineq = self.jacobian_k
    b_ineq = self.jacobian_k * x_k - self.evaluate_constraints(x_k)
    if True:
      print('c')
      print(self.cs[0](x_k))
      print('jacobian')
      print(self.jacobian_k)
      print('jacobian * x_k')
      print(self.jacobian_k * x_k)
      print('jacobian * x_k - constraints')
      print(self.jacobian_k * x_k - self.evaluate_constraints(x_k))
      print('Q')
      print(Q)
      print('A')
      print(A_ineq)
      print('b')
      print(b_ineq)
      print('x')
      print(x_k)

    x, s, pi, f, res, gap, iteration = qp.solve_qp(Q, c, A_eq, b_eq, A_ineq, b_ineq, self.matrix, self.minor_tol)
    assert res < self.minor_tol, "Res = %s, tol = %s" % (res, self.minor_tol)
    assert gap < self.minor_tol, "Gap = %s, tol = %s" % (gap, self.minor_tol)
    print('x after')
    print(x)
    return self.matrix(x), self.matrix(s), self.matrix(pi), iteration

  def _line_search(self, x_k, s_k, pi_k, x_hat, s_hat, pi_hat):
    alpha = mp.one
    def m(x, s, pi):
      cs = self.evaluate_constraints(x)
      phi = self.f(x) + pi.T * (cs - s)
      for j in range(len(self.cs)):
        phi += 0.5 * self.rho[j] * (cs[j] - s[j]) ** 2
      return phi

    m0 = m(x_k, s_k, pi_k)
    for i in range(30):
      x = (mp.one - alpha) * x_k + alpha * x_hat
      s = (mp.one - alpha) * s_k + alpha * s_hat
      pi = (mp.one - alpha) * pi_k + alpha * pi_hat
      phi = m(x, s, pi)
      if self.print_stats:
        print_dps = 10
        phi_str = mp.nstr(phi[0], print_dps)
        m0_str = mp.nstr(m0[0], print_dps)
        print('Value for alpha=%s: %s, m0=%s' % (alpha, phi_str, m0_str))

      if phi[0] < m0[0]:
        return x, s, pi, alpha

      alpha /= 2

    raise ValueError("Current point could not be improved")
