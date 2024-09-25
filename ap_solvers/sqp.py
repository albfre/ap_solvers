from ap_solvers import dense_mp_matrix, qp
from mpmath import mp

class Sqp:
  def __init__(self, f, f_grad, cs, c_grads, tol = mp.mpf('1e-20'), hessian_reset_iter = 50, matrix = dense_mp_matrix.matrix, print_stats = True):
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
    self.matrix = matrix
    self.rho = self.matrix([mp.zero] * len(self.cs)) if self.cs else self.matrix(0)
    self.num_of_rho_norm_trend_switches = 0
    self.previous_rho_norm = 0
    self.tol = tol
    self.minor_tol = tol / 100
    self.print_stats = print_stats
    self.eta = mp.mpf('0.99')
    self.x_used_for_gradient_computation = None
    self.updated_hessian = False
    self.hessian_reset_iter = hessian_reset_iter

  def solve(self, x0, max_iter = 100):
    x_k = self.matrix(x0)
    s_k = -self.evaluate_constraints(x0)
    pi_k = self.matrix(len(self.cs), 1)
    status = ""

    for iteration in range(max_iter):
      if iteration % self.hessian_reset_iter == 0:
        self.reset_hessian(x_k)
      x_hat, s_hat, pi_hat, minor_iterations = self._solve_qp(x_k)
      x_prev, _, _ = x_k, s_k, pi_k
      try:
        x_k, s_k, pi_k, alpha = self._line_search(x_k, s_k, pi_k, x_hat, s_hat, pi_hat)
      except ValueError as e:
        status = "Current point could not be improved"
        alpha = 0.0
        break
      finally:
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

  def reset_hessian(self, x):
    n = len(x)
    self.hessian_approximation = self.matrix(n, n)
    hessian_diagonal = self._finite_difference_grad(self.f, x, True)

    for i in range(n):
      self.hessian_approximation[i, i] = max(hessian_diagonal[i], mp.mpf(self.tol))

  def evaluate_constraints(self, x):
    return self.matrix([c(x) for c in self.cs]) if self.cs else self.matrix(0)

  def _print(self, iteration, x, pi, alpha, minor_iterations):
    print_dps = 2
    obj = mp.nstr(self.f(x), print_dps)
    step = mp.nstr(alpha, print_dps)
    rho = mp.nstr(sum(self.rho) if self.rho.rows > 0 else mp.zero, print_dps)
    c = mp.nstr(min(self.evaluate_constraints(x)) if self.cs else mp.zero, print_dps)

    self._compute_gradients(x)
    dL = mp.nstr(max(abs(xi) for xi in (self._f_grad_k - (self._jacobian_k.T * pi if self.cs else mp.zero))), print_dps)
    if iteration % 10 == 0:
      print("\nIter. \t Step \t Min. \t f(x) \t ||dL(x)|| \t" + "min c".rjust(8) + "\trho \t updated H")
    n = 7
    print(f"{str(iteration).ljust(n)}\t{step.rjust(n)}\t{str(minor_iterations).ljust(n)}\t{obj.rjust(n)}\t{dL.rjust(8)}\t{c.rjust(8)}\t{rho.rjust(n)}\t{str(self.updated_hessian).ljust(n)}")

  def _check_convergence(self, x, pi):
    tau_x = self.tol * (mp.one + max(abs(xi) for xi in x))
    tau_pi = self.tol * (mp.one + max(abs(p) for p in pi)) if self.cs else self.tol

    if any(p < -tau_pi for p in pi): return False
    for i in range(len(self.cs)):
      ci = self.cs[i](x)
      if ci < -tau_x: return False
      if ci * pi[i] > tau_pi: return False

    self._compute_gradients(x)
    d = self._f_grad_k - (self._jacobian_k.T * pi if self.cs else mp.zero)
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
    y = self._f_grad_k - f_grad0 - (deltaJT * pi1 if self.cs else mp.zero)

    p = x_hat - x0 # search direction
    sigma = (alpha * (mp.one - self.eta) * p.T * self.hessian_approximation * p)[0]

    yTdelta = (y.T * delta)[0]
    perform_update = yTdelta >= sigma
    print(yTdelta)
    print(sigma)

    if not perform_update:
      # Second modification in SIAM Review paper on SNOPT
      beta = sigma - yTdelta
      v = deltaJ * delta
      w = self.evaluate_constraints(x1) - self.evaluate_constraints(x0) - jacobian0 * delta if self.cs else self.matrix(0)
      a = self.matrix([vi * wi for vi, wi in zip(v, w)]) if self.cs else [mp.zero]
      if (beta > mp.zero and max(a) > mp.zero) or (beta < mp.zero and min(a) < mp.zero):
        omega = self._solve_identity_hessian_single_constraint_positive_problem(a, beta)
        if (omega.T * omega)[0] < 1e6:
          y = y + deltaJT * self.matrix([oi * wi for oi, wi in zip(omega, w)])
          yTdelta = (y.T * delta)[0]
          perform_update = True

    if perform_update:
      q = self.hessian_approximation * delta
      qTdelta = (q.T * delta)[0]
      self.hessian_approximation += y * y.T / yTdelta - q * q.T / qTdelta
    self.updated_hessian = perform_update

  def _finite_difference_grad(self, f, x, second = False):
    h = self.tol
    grad = self.matrix(len(x), 1)
    x_shift = x.copy()
    if second:
      fx = f(x)

    for i in range(len(x)):
      x_shift[i] = x[i] + h / 2
      fp = f(x_shift)
      x_shift[i] = x[i] - h / 2
      fm = f(x_shift)
      x_shift[i] = x[i]
      grad[i] = (fp - 2 * fx + fm) / h ** 2 if second else (fp - fm) / h
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
    A_ineq = self._jacobian_k if self.cs else []
    b_ineq = self._jacobian_k * x_k - self.evaluate_constraints(x_k) if self.cs else []

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

  def _update_rho(self, x_k, s_k, pi_k, x_hat, s_hat, pi_hat):
    """
    First find the vector rho* that solves
    minimize rho' rho
    subject to phi'(0) = -0.5 p_x' H p_x, rho >= 0,

    where phi(a) is the merit function along the search direction, i.e.,

    phi(a) = f(x + a p_x) - (pi + a p_pi)' (c(x + a p_x) - s - a p_s) + 0.5 sum(rho_i (c_i(x + a p_x) - s_i - a p_s_i)^2)

    The derivative of phi at 0 is

    phi'(0) = g' p_x - p_pi' (c - s) - pi' (J p_x - p_s) + sum(rho_i * (c_i - s_i) * (J_i p_x - p_s_i))

    Using J p_x - p_s = -(c - s), we get

    phi'(0) = g' p_x + (pi - p_pi)' (c - s) - sum(rho_i (c_i - s_i)^2),

    which means that the constraint amounts to

    (c - s).^2 rho = g' p_x + (pi - p_pi)' (c - s) + 0.5 p_x' H p_x
    """
    if not self.cs:
      return

    self._compute_gradients(x_k)
    cs = self.evaluate_constraints(x_k)
    p_x = x_hat - x_k
    p_s = s_hat - s_k
    p_pi = pi_hat - pi_k

    cMinusS = cs - s_k
    pHp = 0.5 * p_x.T * self.hessian_approximation * p_x
    lhs = self.matrix([c**2 for c in cMinusS])
    rhs = (self._f_grad_k.T * p_x + (pi_k - p_pi).T * cMinusS + pHp)[0]

    if rhs > mp.zero and max(lhs) > mp.zero:
      rho_star = self._solve_identity_hessian_single_constraint_positive_problem(lhs, rhs)
      if True:
        delta_rho = 2 ** self.num_of_rho_norm_trend_switches
        for i in range(len(self.rho)):
          rho_hat = self.rho[i] if self.rho[i] <= 4 * (rho_star[i] + delta_rho) else mp.sqrt(self.rho[i] * (rho_star[i] + delta_rho))
          self.rho[i] = max(rho_hat, rho_star[i])

        rho_norm = (self.rho.T * self.rho)[0]
        is_rho_norm_increasing = self.num_of_rho_norm_trend_switches % 2 == 0
        if (rho_norm < self.previous_rho_norm and is_rho_norm_increasing) or (rho_norm > self.previous_rho_norm and not is_rho_norm_increasing):
          self.num_of_rho_norm_trend_switches += 1
        self.previous_rho_norm = rho_norm
      else:
        self.rho = rho_star

  def _line_search(self, x_k, s_k, pi_k, x_hat, s_hat, pi_hat):
    self._update_rho(x_k, s_k, pi_k, x_hat, s_hat, pi_hat)
    mu = mp.mpf('1e-4')
    p_x, p_s, p_pi = x_hat - x_k, s_hat - s_k if self.cs else self.matrix(0), pi_hat - pi_k if self.cs else self.matrix(0)

    def trial_point(alpha):
      return x_k + alpha * p_x, s_k + alpha * p_s if self.cs else self.matrix(0), pi_k + alpha * p_pi if self.cs else self.matrix(0)
      
    def merit_function(alpha):
      x, s, pi = trial_point(alpha)
      if not self.cs:
        return self.f(x)

      cs = self.evaluate_constraints(x)
      phi = self.f(x) - pi.T * (cs - s)
      for j in range(len(cs)):
        phi += 0.5 * self.rho[j] * (cs[j] - s[j]) ** 2
      return phi[0]

    def merit_function_derivative(alpha):
      #phi(a) = f(x + a p_x) - (pi + a p_pi)' (c(x + a p_x) - s - a p_s) + 0.5 sum(rho_i (c_i(x + a p_x) - s_i - a p_s_i)^2)
      #phi'(a) = g(x + a p_x)' p_x - p_pi' (c(x + a p_x) - s - a p_s) - (pi + a p_pi)' (J(x + a p_x) p_x - p_s) + sum(rho_i (c_i(x + a p_x) - s_i - a p_s_i) (J(x + a p_x) p_x - p_s_i)
      x, s, pi = trial_point(alpha)
      self._compute_gradients(x)
      v = (self._f_grad_k.T * p_x)[0]

      if self.cs:
        cs = self.evaluate_constraints(x)
        jacPxMinusPs = self._jacobian_k * p_x - p_s
        v -= (p_pi.T * (cs - s) + pi.T * jacPxMinusPs)[0]
        for j in range(len(cs)):
          v += self.rho[j] * (cs[j] - s[j]) * jacPxMinusPs[j]
      return v
      
    self._compute_gradients(x_k)
    cs = self.evaluate_constraints(x_k)

    m_0 = merit_function(mp.zero)
    d_m_0 = merit_function_derivative(mp.zero)

    def psi(alpha):
      m_alpha = merit_function(alpha)
      return m_alpha - m_0 - mu * alpha * d_m_0

    #cs_k = self.evaluate_constraints(x_k)
    #tau_v = mp.mpf('10')
    #b = tau_v * self.matrix([max(mp.one, c) for c in cs_k])

    alpha = mp.one
    eta = 0.4
    for i in range(30):
      x, s, pi = trial_point(alpha)

      #cs = self.evaluate_constraints(x)
      #bounded_constraint_violations = all(c >= -bi for c, bi in zip(cs, b))

      if False:
        print_dps = 10
        print(mp.nstr(alpha, print_dps))
        print(mp.nstr(-eta * d_m_0, print_dps))
        print(mp.nstr(abs(merit_function_derivative(alpha)), print_dps))
        print(mp.nstr(psi(alpha), print_dps))
      if psi(alpha) < mp.zero: # and abs(merit_function_derivative(alpha)) <= eta * abs(d_m_0):
        cs = self.evaluate_constraints(x)
        s = self.matrix([si if rho_i == mp.zero else max(ci + pi_i / rho_i, mp.zero) for si, ci, pi_i, rho_i in zip(s, cs, pi, self.rho)]) if self.cs else self.matrix(0)
        return x, s, pi, alpha

      alpha *= 0.5

    raise ValueError("Current point could not be improved")
