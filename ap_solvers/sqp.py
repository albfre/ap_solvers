from ap_solvers import dense_mp_matrix, qp
from mpmath import mp

class Sqp:
  def __init__(self, f, f_grad, cs, c_grads, tol = mp.mpf('1e-20'), matrix = dense_mp_matrix.matrix, print_stats = True):
    """
    Initialize the Sqp object with a set of points.

    minimize f(x)
    s.t. c_i(x) >= 0 for all i

    if f_grad or c_grad is None, they will be approximated by finite differences
    """
    self.f = f
    self.f_grad = f_grad
    self.cs = cs if cs else []
    self.c_grads = c_grads if c_grads else []
    self.rho = [mp.zero] * len(self.cs)
    self.tol = tol
    self.minor_tol = tol / 100
    self.matrix = matrix
    self.print_stats = print_stats
    self.eta = mp.mpf('0.5')
    self.x_used_for_gradient_computation = None
    self.updated_hessian = False

  def solve(self, x0, max_iter = 100):
    n = len(x0)
    self.hessian_approximation = self.matrix(n, n)
    for i in range(n):
      self.hessian_approximation[i, i] = mp.one

    x_k = self.matrix(x0)
    s_k = self.matrix(len(self.cs), 1)
    pi_k = self.matrix(len(self.cs), 1)
    status = ""

    for iteration in range(max_iter):
      x_hat, s_hat, pi_hat, minor_iterations = self._solve_qp(x_k)
      x_prev, s_prev, pi_prev = x_k, s_k, pi_k
      try:
        x_k, s_k, pi_k, alpha = self._line_search(x_k, s_k, pi_k, x_hat, s_hat, pi_hat)
      except ValueError as e:
        status = "Current point could not be improved"
        break

      if self.print_stats:
        self._print(iteration, x_k, pi_k, alpha, minor_iterations)

      if self._check_convergence(x_k, pi_k):
        status = "Optimal solution found"
        break

      if iteration + 1 < max_iter:
        self._update_hessian_approximation(x_prev, x_k, x_hat, pi_k, alpha)

    if iteration + 1 == max_iter:
      status = "Maximum number of iterations"

    return x_k, self.f(x_k), self.evaluate_constraints(x_k), status

  def evaluate_constraints(self, x):
    return self.matrix([c(x) for c in self.cs])

  def _print(self, iteration, x, pi, alpha, minor_iterations):
    print_dps = 2
    obj = mp.nstr(self.f(x), print_dps)
    step = mp.nstr(alpha, print_dps)
    rho = mp.nstr(sum(self.rho), print_dps)
    c = mp.nstr(min(self.evaluate_constraints(x)), print_dps)

    #print('grad: ' + str(self._f_grad_k))
    #print('J pi: ' + str(self._jacobian_k.T * pi))
    d = mp.nstr(max(abs(xi) for xi in (self._f_grad_k - self._jacobian_k.T * pi)), print_dps)
    if iteration % 10 == 0:
      print("Iter. \t Step \t Min. \t f(x) \t ||dL(x)|| \t" + "min c".rjust(8) + "\trho \t updated H")
    n = 7
    print(f"{str(iteration).ljust(n)}\t{step.rjust(n)}\t{str(minor_iterations).ljust(n)}\t{obj.rjust(n)}\t{d.rjust(8)}\t{c.rjust(8)}\t{rho.rjust(n)}\t{str(self.updated_hessian).ljust(n)}")
    #print('%s \t %s \t %s \t %s \t %s \t %s \t %s \t %s' % (iteration, step, minor_iterations, obj, d, c, rho, self.updated_hessian))

  def _check_convergence(self, x, pi):
    tau_x = self.tol * (mp.one + max(abs(xi) for xi in x))
    tau_pi = self.tol * (mp.one + max(abs(p) for p in pi))

    if any(p < -tau_pi for p in pi): return False
    for i in range(len(self.cs)):
      ci = self.cs[i](x)
      if ci < -tau_x: return False
      if ci * pi[i] > tau_pi: return False

    self._compute_gradients(x)
    d = self._f_grad_k - self._jacobian_k.T * pi
    if any(abs(di) > tau_pi for di in d): return False
    return True


  def _update_hessian_approximation(self, x0, x1, x_hat, pi1, alpha):
    self._compute_gradients(x0)
    f_grad0 = self._f_grad_k
    jacobian0 = self._jacobian_k

    self._compute_gradients(x1)
    delta = x1 - x0
    deltaJ = self._jacobian_k - jacobian0
    deltaJT = deltaJ.T
    y = self._f_grad_k - f_grad0 - deltaJT * pi1

    p = x_hat - x0 # search direction
    sigma = (alpha * (mp.one - self.eta) * p.T * self.hessian_approximation * p)[0]

    yTdelta = (y.T * delta)[0]
    perform_update = yTdelta >= sigma

    if not perform_update:
      # Second modification in SIAM Review paper on SNOPT
      beta = sigma - yTdelta
      v = deltaJ * delta
      w = self.evaluate_constraints(x1) - self.evaluate_constraints(x0) - jacobian0 * delta
      a = self.matrix([vi * wi for vi, wi in zip(v, w)])
      if (beta > mp.zero and max(a) > mp.zero) or (beta < mp.zero and min(a) < mp.zero):
        omega = self._solve_identity_hessian_single_constraint_positive_problem(a, beta)
        if (omega.T * omega)[0] < 1e6:
          y = y + deltaJT * self.matrix([oi * wi for oi, wi in zip(omega, w)])
          yTdelta = (y.T * delta)[0]
          perform_update = True

    if perform_update:
      q = self.hessian_approximation * delta
      qTdelta = (q.T * delta)[0]
      self.hessian_approximation += y * y.T / yTdelta + q * q.T / qTdelta
    self.updated_hessian = perform_update

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
    if self.x_used_for_gradient_computation and x == self.x_used_for_gradient_computation:
      return

    self._f_grad_k = self.f_grad(x) if self.f_grad else self._finite_difference_grad(self.f, x)
    self._jacobian_k = self.matrix(len(self.cs), len(x))
    for i in range(len(self.cs)):
      if self.c_grads and self.c_grads[i]:
        grad = self.c_grads[i](x)
      else:
        grad = self._finite_difference_grad(self.cs[i], x)
      for j in range(len(grad)):
        self._jacobian_k[i, j] = grad[j]
    self.x_used_for_gradient_computation = x
    
  def _solve_qp(self, x_k):
    """ minimize f_k + g_k'(x - x_k) + 0.5 (x - x_k)' H_k (x - x_k)
        s.t. c_k + J_k (x - x_k) >= 0

        i.e., 

        minimize (g_k - H_k x_k)' x + 0.5 x' H_k x + constants
        s.t. J_k x >= J_k x_k - c_k
    """
    Q = self.hessian_approximation
    
    self._compute_gradients(x_k)
    c = self._f_grad_k - Q * x_k
    A_eq = []
    b_eq = []
    A_ineq = self._jacobian_k
    b_ineq = self._jacobian_k * x_k - self.evaluate_constraints(x_k)

    x, s, pi, f, res, gap, iteration = qp.solve_qp(Q, c, A_eq, b_eq, A_ineq, b_ineq, self.matrix, self.minor_tol, 100, False)
    assert res < self.minor_tol, "Res = %s, tol = %s" % (res, self.minor_tol)
    assert gap < self.minor_tol, "Gap = %s, tol = %s" % (gap, self.minor_tol)
    return self.matrix(x), self.matrix(s), self.matrix(pi), iteration

  def _solve_identity_hessian_single_constraint_positive_problem(self, a, b):
    """
    minimize x' x
    subject to a'x = b, x >= 0

    If b >= 0, the solution is 
    x_i = (b / (ap' ap)) * a_i if a_i > 0, and 0 otherwise,
    where, ap is the positive elements of a
    """
    if b < mp.zero:
      a, b = -a, -b
    ap_norm = sum(ap * ap for ap in a if ap > mp.zero)
    return mp.matrix([max((b / ap_norm) * ai, mp.zero) for ai in a])

  def _update_rho(self, x_k, s_k, pi_k, x_hat, pi_hat):
    """
    First find the vector rho* that solves
    minimize rho' rho
    subject to phi'(0) = -0.5 p_x' H p_x, rho >= 0,

    where phi(a) is the merit function along the search direction, i.e.,

    phi(a) = f(x + a p_x) - (pi + a p_pi)' (c(x + a p_x) - s - a p_s) + 0.5 sum(rho_i (c_i(x + a p_x) - s_i)^2)

    The derivative of phi at 0 is

    phi'(0) = g' p_x - p_pi' (c - s) - pi' (J p_x - p_s) + sum(rho_i * (c_i - s_i) * (J_i p_x - p_s_i))

    Using J p_x - p_s = -(c - s), we get

    phi'(0) = g' p_x + (pi - p_pi)' (c - s) - sum(rho_i (c_i - s_i)^2),

    which means that the constraint amounts to

    (c - s).^2 rho = g' p_x + (pi - p_pi)' (c - s) + 0.5 p_x' H p_x
    """
    self._compute_gradients(x_k)
    cs = self.evaluate_constraints(x_k)
    p_x = x_hat - x_k
    p_pi = pi_hat - pi_k
    cMinusS = cs - s_k
    cMinusS2 = self.matrix([c**2 for c in cMinusS])
    rhs = (self._f_grad_k.T * p_x + (pi_k - p_pi).T * cMinusS + 0.5 * p_x.T * self.hessian_approximation * p_x)[0]
    if rhs > mp.zero and max(cMinusS2) > mp.zero:
      rho_star = self._solve_identity_hessian_single_constraint_positive_problem(cMinusS2, rhs)

      if True:
        delta_rho = mp.one
        rho_hat = self.rho.copy()
        for i in range(len(self.rho)):
          if self.rho[i] > 4 * (rho_star[i] + delta_rho):
            rho_hat[i] = mp.sqrt(self.rho[i] * (rho_star[i] + delta_rho))
        for i in range(len(self.rho)):
          self.rho[i] = max(rho_hat[i], rho_star[i])
      else:
        self.rho = rho_star

  def _line_search(self, x_k, s_k, pi_k, x_hat, s_hat, pi_hat):
    self._update_rho(x_k, s_k, pi_k, x_hat, pi_hat)

    alpha = mp.one
    def merit_function(x, s, pi):
      cs = self.evaluate_constraints(x)
      phi = self.f(x) - pi.T * (cs - s)
      for j in range(len(cs)):
        phi += 0.5 * self.rho[j] * (cs[j] - s[j]) ** 2
      return phi

    m0 = merit_function(x_k, s_k, pi_k)
    for i in range(30):
      x = (mp.one - alpha) * x_k + alpha * x_hat
      s = (mp.one - alpha) * s_k + alpha * s_hat
      pi = (mp.one - alpha) * pi_k + alpha * pi_hat
      phi = merit_function(x, s, pi)

      if phi[0] < m0[0]:
        return x, s, pi, alpha

      alpha /= 2

    raise ValueError("Current point could not be improved")
