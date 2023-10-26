from mpmath import mp

def symmetric_indefinite_factorization(A_in):
  A = mp.matrix(A_in)
  assert(A.rows == A.cols)
  n = A.rows
  alpha = (mp.one + mp.sqrt(mp.mpf(17))) / mp.mpf(8)
  ipiv = [mp.zero] * n

  def max_in_row_or_column(i_begin, i_end, constant_index, check_column):
    i_max = 0
    col_max = mp.mpf(0.0)
    for i in range(i_begin, i_end):
      v = abs(A[i, constant_index] if check_column else A[constant_index, i])
      if v > col_max:
        col_max = v
        i_max = i
    return i_max, col_max

  info = 0
  k = 0
  while k < n:
    k_step = 1
    kp = 0
    absakk = abs(A[k, k])
    # imax is the row-index of the largest off-diagonal element in column k, and colmax is its absolute value
    i_max, col_max = max_in_row_or_column(k + 1, n, k, True)
    if (absakk == mp.zero and col_max == mp.zero):
      # Column k is zero: set info and continue
      if info == 0:
        info = k
        kp = k
    else:
      if absakk >= alpha * col_max:
        # no interchange, use 1-by-1 pivot block
        kp = k
      else:
        # jmax is the column-index of the largest off-diagonal element in row imax, and rowmax is its absolute value
        j_max1, row_max1 = max_in_row_or_column(k, i_max, i_max, False)
        j_max2, row_max2 = max_in_row_or_column(i_max + 1, n, i_max, True)
        row_max = max(row_max1, row_max2)
        if absakk * row_max >= alpha * col_max * col_max:
          kp = k # no interchange, use 1-by-1 pivot block
        elif abs(A[i_max, i_max]) >= alpha * row_max:
          kp = i_max # interchange rows and columns k and imax, use 1-by-1 pivot block
        else:
          # interchange rows and columns k+1 and imax, use 2-by-2 pivot block
          kp = i_max
          k_step = 2

      kk = k + k_step - 1
      if kp != kk:
        # Interchange rows and columns kk and kp in the trailing submatrix A(k:n,k:n)
        for i in range(kp + 1, n):
          A[i, kp], A[i, kk] = A[i, kk], A[i, kp]

        for j in range(kk + 1, kp):
          A[kp, j], A[j, kk] = A[j, kk], A[kp, j]

        A[kp, kp], A[kk, kk] = A[kk, kk], A[kp, kp]

        if k_step == 2:
          A[kk, k], A[kp, k] = A[kp, k], A[kk, k]

      # Update the trailing submatrix
      if k_step == 1:
        # 1-by-1 pivot block D(k): column k now holds W(k) = L(k)*D(k) where L(k) is the k-th column of L
        # Perform a rank-1 update of A(k+1:n,k+1:n) as A := A - L(k)*D(k)*L(k)**T = A - W(k)*(1/D(k))*W(k)**T
        r1 = mp.one / A[k, k]
        for j in range(k + 1, n):
          r1ajk = r1 * A[j, k]
          for i in range(j, n):
            A[i, j] -= r1ajk * A[i, k]

        for i in range(k + 1, n):
          A[i, k] *= r1

      else:
        # 2-by-2 pivot block D(k): columns k and k+1 now hold ( W(k) W(k+1) ) = ( L(k) L(k+1) )*D(k)
        # where L(k) and L(k+1) are the k-th and (k+1)-th columns of L
        if k < n - 1:
          # Perform a rank-2 update of A(k+2:n,k+2:n) as
          # A := A - ( L(k) L(k+1) )*D(k)*( L(k) L(k+1) )**T = A - ( W(k) W(k+1) )*inv(D(k))*( W(k) W(k+1) )**T
          # where L(k) and L(k+1) are the k-th and (k+1)-th columns of L

          d21 = A[k + 1, k]
          d11 = A[k + 1, k + 1] / d21
          d22 = A[k, k] / d21
          t = mp.one / (d11 * d22 - mp.one)
          d21 = t / d21

          for j in range(k + 2, n):
            wk = d21 * (d11 * A[j, k] - A[j, k + 1])
            wkp1 = d21 * (d22 * A[j, k + 1] - A[j, k])
            for i in range(j, n):
              A[i, j] -= (A[i, k] * wk + A[i, k + 1] * wkp1)

            A[j, k] = wk
            A[j, k + 1] = wkp1

    # Store details of the interchanges in ipiv
    if k_step == 1:
      ipiv[k] = kp
    else:
      ipiv[k] = -kp
      ipiv[k + 1] = -kp

    k += k_step

  return A, ipiv, info


def solve_using_factorization(L, ipiv, b_in):
  b = mp.matrix(b_in)
  # Solve A*X = B, where A = L*D*L**T.
  n = b.rows

  # First solve L*D*X = B, overwriting B with X.
  # k is the main loop index, increasing from 1 to n in steps of 1 or 2, depending on the size of the diagonal blocks.
  def dger(i_start, j_index):
    temp = -b[j_index]
    for i in range(i_start, n):
      b[i] += L[i, j_index] * temp

  k = 0
  while k < n:
    if ipiv[k] >= 0:
      # 1 x 1 diagonal block, interchange rows k and ipiv(k).
      kp = ipiv[k]
      if kp != k:
        b[k], b[kp] = b[kp], b[k]

      # Multiply by inv(L(k)), where L(k) is the transformation stored in column k of L.
      dger(k + 1, k)
      b[k] /= L[k, k]
      k += 1
    
    else:
      # 2 x 2 diagonal block, interchange rows k+1 and -ipiv(k).

      kp = -ipiv[k]
      if kp != k + 1:
        b[k + 1], b[kp] = b[kp], b[k + 1]

      # Multiply by inv(L(k)), where L(k) is the transformation stored in columns k and k+1 of L.
      if k < n - 1:
        dger(k + 2, k)
        dger(k + 2, k + 1)

      # Multiply by the inverse of the diagonal block.
      akm1k = L[k + 1, k]
      akm1 = L[k, k] / akm1k
      ak = L[k + 1, k + 1] / akm1k
      denom = akm1 * ak - mp.one
      bkm1 = b[k] / akm1k
      bk = b[k + 1] / akm1k
      b[k] = (ak * bkm1 - bk) / denom
      b[k + 1] = (akm1 * bk - bkm1) / denom
      k = k + 2

  # Next solve L**T *X = B, overwriting B with X.
  # k is the main loop index, decreasing from n-1 to 0 in steps of 1 or 2, depending on the size of the diagonal blocks.
  def dgemv(i_start, j_index):
    temp = mp.zero
    for i in range(i_start, n):
      temp += L[i, j_index] * b[i]
    b[j_index] -= temp

  k = n - 1
  while k >= 0:
    if ipiv[k] >= 0:
      # 1 x 1 diagonal block, multiply by inv(L**T(k)), where L(k) is the transformation stored in column k of L.
      if k < n - 1:
        dgemv(k + 1, k) # Subroutine dgemv 'Transpose' with alpha = -1 and beta = 1

      #c Interchange rows K and IPIV(K).
      kp = ipiv[k]
      if kp != k:
        b[k], b[kp] = b[kp], b[k]

      k -= 1

    else:
      # 2 x 2 diagonal block, multiply by inv(L**T(k-1)), where L(k-1) is the transformation stored in columns k-1 and k of L.
      if k < n - 1:
        dgemv(k + 1, k)
        dgemv(k + 1, k - 1)

      # Interchange rows k and -ipiv(k).
      kp = -ipiv[k]
      if kp != k:
        b[k], b[kp] = b[kp], b[k]

      k -= 2

  return b
