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

  def solve(self, x0, max_iter = 100):
    n = len(x0)
    self.hessian_approximation = self.matrix(n, n)
    for i in range(n):
      self.hessian_approximation[i, i] = mp.one

    x_k = self.matrix(x0)
    s_k = self.matrix(len(self.cs), 1)
    pi_k = self.matrix(len(self.cs), 1)
    self._compute_gradients(x_k)

    for iteration in range(max_iter):
      x_bar, s_bar, pi_bar = self._solve_qp(x_k)
      x_next, s_next, pi_next = self._line_search(x_k, s_k, pi_k, x_bar, s_bar, pi_bar)

      f_grad_prev = self.f_grad_k.copy()
      jacobian_prev = self.jacobian_k.copy()

      self._compute_gradients(x_next)
      delta_k = x_next - x_k
      y_k = self.f_grad_k - f_grad_prev - (self.jacobian_k - jacobian_prev).T * pi_next

  def _finite_difference_grad(self, f, x):
    h = self.tol
    grad = self.matrix(len(x), 1)
    x_shift = self.matrix(x)
    for i in range(len(x)):
      x_shift[i] = x[i] + h
      fp = f(x_shift)
      x_shift[i] = x[i] - h
      fm = f(x_shift)
      x_shift[i] = x[i]
      grad[i] = (fp - fm) / (2 * h)
    return grad

  def _compute_gradients(self, x):
    self.f_grad_k = self.f_grad(x) if self.f_grad else self._finite_difference_grad(self.f, x)
    self.jacobian_k = self.matrix(len(self.cs), len(x))
    for i in range(len(cs)):
      if self.c_grads and self.c_grads[i]:
        grad = self.c_grads[i](x)
      else:
        grad = self._finite_difference_grad(self.cs[i], x)
      for j in range(len(grad)):
        self.jacobian_k[i, j] = grad[j]
    
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
    A_ineq = self.jacobian_k
    b_ineq = self.jacobian_k * x_k - self.constraints(x_k)_k

    x, s, pi, f, res, gap, iteration = qp.solve_qp(Q, c, A_eq, b_eq, A_ineq, b_ineq, self.matrix, self.minor_tol)
    assert res < self.minor_tol, "Res = %s, tol = %s" % (res, self.minor_tol)
    assert gap < self.minor_tol, "Gap = %s, tol = %s" % (gap, self.minor_tol)
    return self.matrix(x), self.matrix(s), self.matrix(pi)

    def _line_search(self, x_k, s_k, pi_k, x_bar, s_bar, pi_bar):
      alpha = mp.one
      def m(x, s, pi):
        cs = self.constraints(x)
        phi = self.f(x) - pi.T * (cs - s)
        for j in range(len(self.cs)):
          phi += 0.5 * self.rho[j] * (cs[j] - s[j]) ** 2
        return phi

      m_0 = m(x_k, s_k, pi_k)
      for i in range(10):
        x = (mp.one - alpha) * x_k + alpha * x_bar
        s = (mp.one - alpha) * s_k + alpha * s_bar
        pi = (mp.one - alpha) * pi_k + alpha * pi_bar
        phi = m(x, s, pi)

        if phi < m_0:
          return x, s, pi

      raise ValueError("Current point could not be improved")
