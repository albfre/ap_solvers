from mpmath import mp
from src import bunch_kaufman

def solve_qp(H, c, A_eq, b_eq, A_ineq, b_ineq, matrix=mp.matrix, tol=mp.mpf('1e-20'), max_iter=100):
  """ minimize 0.5 x' H x + c' x
      st    Aeq x = beq
            Aineq x >= bineq
  """
  H = matrix(H)
  c = matrix(c)
  A_eq = matrix(A_eq)
  b_eq = matrix(b_eq)
  A_ineq = matrix(A_ineq)
  b_ineq = matrix(b_ineq)

  n = H.rows
  m_ineq = A_ineq.rows
  m_eq = A_eq.rows

  assert(H.rows == H.cols)
  assert(A_ineq.cols == n)
  assert(A_eq.cols == n)
  assert(b_ineq.rows == m_ineq)
  assert(b_eq.rows == m_eq)

  A_ineq_T = A_ineq.T
  A_eq_T = A_eq.T
  c_T = c.T

  # Define the function for computing the residual rS
  def eval_r_s(s, z, mu):
    return elementwise_product(s, z) - mu # SZe - mu e

  # Define the function for evaluating the objective and constraints
  def eval_func(x, s, y, z, mu):
    Hx = H * x
    A_eqx = A_eq * x
    A_ineqx = A_ineq * x

    # Objective
    f = (0.5 * x.T * Hx + c_T * x)[0] # 0.5 x' H x + c' x

    # Residuals
    r_grad = Hx + c # Hx + c + Aeq' y - Aineq' z
    if m_eq > 0:
      r_grad += A_eq_T * y
    if m_ineq > 0:
      r_grad -= A_ineq_T * z

    r_eq = A_eqx - b_eq
    r_ineq = A_ineqx - s - b_ineq
    r_s = eval_r_s(s, z, mu) # SZe - mu e

    return f, r_grad, r_eq, r_ineq, r_s

  # Construct the augmented KKT system
  """ [ H       Aeq'   Aineq' ]
      [ Aeq      0      0     ]
      [ Aineq    0   -Z^-1 S  ]
  """
  m = n + m_eq + m_ineq
  KKT = matrix(m, m)
  set_submatrix(KKT, H, 0, 0)
  set_submatrix(KKT, A_eq_T, 0, n)
  set_submatrix(KKT, A_ineq_T, 0, n + m_eq)
  set_submatrix(KKT, A_eq, n, 0)
  set_submatrix(KKT, A_ineq, n + m_eq, 0)

  def update_matrix(s, z):
    minusZinvS = -elementwise_division(s, z)
    set_subdiagonal(KKT, minusZinvS, n + m_eq, n + m_eq)

  # Define the function for computing the search direction
  def compute_search_direction(s, z, L, ipiv, r_grad, r_eq, r_ineq, r_s):
    r_ineqMinusYinvrS = r_ineq + elementwise_division(r_s, z) # Aineq x - s - bineq + Z^-1 (SZe - mue)
    rhs = -matrix(r_grad.tolist() + r_eq.tolist() + r_ineqMinusYinvrS.tolist())

    # Solve the KKT system
    d = bunch_kaufman.overwriting_solve_using_factorization(L, ipiv, rhs)

    # Extract the search direction components
    d = d.tolist()
    dx = matrix(d[:n])
    dy = matrix(d[n:n + m_eq])
    dz = -matrix(d[n + m_eq:n + m_eq + m_ineq])
    ds = -elementwise_division(r_s + elementwise_product(s, dz), z) # -Z^-1 (rS + S dz)

    return dx, ds, dy, dz

  # Define the function for computing the step size
  def get_max_step(v, dv):
    assert(v.cols == 1)
    return min(min(-v[i] / dv[i] if dv[i] < 0 else 1 for i in range(v.rows)), mp.one)

  # Initialize primal and dual variables
  x = matrix([1] * n)      # Primal variables
  s = matrix([1] * m_ineq) # Slack variables for inequality constraints
  y = matrix([1] * m_eq)   # Multipliers for equality constraints
  z = matrix([1] * m_ineq) # Multipliers for inequality constraints
  
  def get_mu(s, z):
    return (s.T * z / m_ineq)[0] if m_ineq > 0 else mp.zero

  def get_residual_and_gap(s, z, r_grad, r_eq, r_ineq):
    res = mp.norm(mp.matrix(r_grad.tolist() + r_eq.tolist() + r_ineq.tolist()))
    #res = mp.norm(r_grad) + mp.norm(r_eq) + mp.norm(r_ineq)
    gap = get_mu(s, z)
    return res, gap

  # Perform the interior point optimization
  for iteration in range(max_iter):
    f, r_grad, r_eq, r_ineq, r_s = eval_func(x, s, y, z, 0)

    # Check the convergence criterion
    res, gap = get_residual_and_gap(s, z, r_grad, r_eq, r_ineq)
    print_dps = 10
    print('%s. f: %s, res: %s, gap: %s' % (iteration, mp.nstr(f, print_dps), mp.nstr(res, print_dps), mp.nstr(gap, print_dps)))
    if res <= tol and gap <= tol:
      break

    # Update and factorize KKT matrix
    update_matrix(s, z)
    L, ipiv, info = bunch_kaufman.overwriting_symmetric_indefinite_factorization(KKT.copy())
    assert(info == 0)

    # Use the predictor-corrector method

    # Compute affine scaling step
    dx_aff, ds_aff, dy_aff, dz_aff = compute_search_direction(s, z, L, ipiv, r_grad, r_eq, r_ineq, r_s)
    alpha_aff_p = get_max_step(s, ds_aff)
    alpha_aff_d = get_max_step(z, dz_aff)
    s_aff = matrix(s) + alpha_aff_p * ds_aff
    z_aff = matrix(z) + alpha_aff_d * dz_aff
    mu_aff = get_mu(z_aff, s_aff)

    # Compute aggregated centering-corrector direction
    mu = get_mu(s, z)
    sigma = (mu_aff / mu) ** 3.0 if mu > mp.zero else 0
    r_s_center = eval_r_s(s, z, sigma * mu)
    r_s_center_corr = elementwise_product(dz_aff, ds_aff) + r_s_center
    dx, ds, dy, dz = compute_search_direction(s, z, L, ipiv, r_grad, r_eq, r_ineq, r_s_center_corr)
    alpha_p = get_max_step(s, ds)
    alpha_d = get_max_step(z, dz)

    # Update the variables
    fraction_to_boundary = 0.995
    x += fraction_to_boundary * alpha_p * dx
    s += fraction_to_boundary * alpha_p * ds
    y += fraction_to_boundary * alpha_d * dy
    z += fraction_to_boundary * alpha_d * dz

  # Return the solution and objective value
  f, r_grad, r_eq, r_ineq, r_s = eval_func(x, s, y, z, 0)
  res, gap = get_residual_and_gap(s, z, r_grad, r_eq, r_ineq)
  return x, f, res, gap, iteration

# Helper functions for linear algebra operations

def elementwise_product(x, y):
  r = x.copy()
  for i in range(x.rows):
    for j in range(x.cols):
      r[i, j] *= y[i, j]
  return r

def elementwise_division(x, y):
  r = x.copy()
  for i in range(x.rows):
    for j in range(x.cols):
      r[i, j] /= y[i, j]
  return r

def set_submatrix(M, X, start_i, start_j):
  m = X.rows
  n = X.cols
  assert(M.rows >= m + start_i)
  assert(M.cols >= n + start_j)
  for i in range(m):
    si = i + start_i
    for j in range(n):
      M[si, j + start_j] = X[i, j]

def set_subdiagonal(M, d, start_i, start_j):
  assert(d.cols == 1)
  m = d.rows
  assert(M.rows >= m + start_i)
  assert(M.cols >= m + start_j)
  for i in range(m):
    M[i + start_i, i + start_j] = d[i];
