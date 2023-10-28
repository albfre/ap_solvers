from mpmath import mp

class matrix:
  def __init__(self, *args):
    if isinstance(args[0], (list, tuple)):
      if isinstance(args[0][0], (list, tuple)):
        # nested row matrix
        self.rows = len(args[0])
        self.cols = len(args[0][0])
        self.__data = [[mp.mpf(x) for x in row] for row in args[0]]
      else:
        # row vector
        v = args[0]
        self.rows = len(v)
        self.cols = 1
        self.__data = [[mp.mpf(x)] for x in v]
    elif isinstance(args[0], int):
      # empty matrix of given dimensions
      if len(args) == 1:
        self.rows = self.cols = args[0]
      else:
        if not isinstance(args[1], int):
          raise TypeError("Expected int")
        self.rows = args[0]
        self.cols = args[1]
      self.__data = [[mp.zero for _ in range(self.cols)] for _ in range(self.rows)]
    elif isinstance(args[0], (mp.matrix, matrix)):
      A = args[0]
      self.rows = A.rows
      self.cols = A.cols
      if isinstance(A, mp.matrix):
        self.__data = [[A[i, j] for j in range(self.cols)] for i in range(self.rows)]
      else:
        self.__data = A.__data.copy()
    else:
      raise TypeError("Could not interpret given arguments")

  def __getitem__(self, index):
    if isinstance(index, int):
      if self.rows == 1:
        return self.__data[0][index]
      elif self.cols == 1:
        return self.__data[index][0]
      else:
        raise TypeError("Invalid index format")

    if isinstance(index, tuple) and len(index) == 2:
      row, col = index
      return self.__data[row][col]
    else:
      raise TypeError("Invalid index format")

  def __setitem__(self, index, value):
    if isinstance(index, int):
      if self.rows == 1:
        self.__data[0][index] = value
      elif self.cols == 1:
        self.__data[index][0] = value
      else:
        raise TypeError("Invalid index format: " + str(index) + " " + str(type(index)))
    elif isinstance(index, tuple) and len(index) == 2:
      row, col = index
      self.__data[row][col] = value
    else:
      raise TypeError("Invalid index format: " + str(index) + " " + str(type(index)))

  def _verify_size(self, other):
    if (self.cols != other.cols or self.rows != other.rows):
      raise ValueError("Dimensions do not match")

  def _elementwise(self, other, operation):
    result = matrix(0, 0)
    result.rows = self.rows
    result.cols = self.cols
    if isinstance(other, matrix):
      self._verify_size(other)
      result.__data = [[operation(x, y) for x, y in zip(self.__data[i], other.__data[i])] for i in range(self.rows)]

    else:
      result.__data = [[operation(x, other) for x in row] for row in self.__data]
    return result

  def __str__(self):
    return "\n".join(" ".join(str(self[i, j]) for j in range(self.cols)) for i in range(self.rows))

  def __mul__(self, other):
    if isinstance(other, matrix):
      # matrix multiplication
      if self.cols != other.rows:
        raise ValueError("Dimensions not compatible for multiplication")

      result = matrix(self.rows, other.cols)
      for i in range(self.rows):
        for j in range(other.cols):
          for k in range(self.cols):
            result[i, j] += self[i, k] * other[k, j]
      return result
    else:
      # scalar multiplication
      return self._elementwise(other, lambda x, y: x * y)

  def __rmul__(self, other):
    if isinstance(other, matrix):
      raise TypeError("Other should not be of type matrix")
    return self.__mul__(other)

  def __neg__(self):
    return (-1) * self

  def __div__(self, other):
    assert(not isinstance(other, matrix))
    return (mp.one / other) * self

  __truediv__ = __div__

  def __add__(self, other):
    return self._elementwise(other, lambda x, y: x + y)

  def __radd__(self, other):
    return self.__add__(other)

  def __sub__(self, other):
    return self._elementwise(other, lambda x, y: x - y)

  def __rsub__(self, other):
    return -self + other

  def __len__(self):
    if self.rows == 1:
      return self.cols
    elif self.cols == 1:
      return self.rows
    else:
      return self.rows # do like numpy

  def transpose(self):
    result = matrix(self.cols, self.rows)
    for i in range(self.rows):
      for j in range(self.cols):
        result[j, i] = self[i, j]
    return result

  T = property(transpose)

  def tolist(self):
    return self.__data

  def copy(self):
    return matrix(self.__data)
