"""Tests for the dense_mp_matrix module."""

import unittest
from mpmath import mp
from ap_solvers.dense_mp_matrix import matrix


class TestMatrixConstruction(unittest.TestCase):
    """Tests for matrix construction."""

    def test_from_nested_list(self):
        m = matrix([[1, 2], [3, 4]])
        self.assertEqual(m.rows, 2)
        self.assertEqual(m.cols, 2)
        self.assertEqual(m[0, 0], mp.mpf(1))
        self.assertEqual(m[1, 1], mp.mpf(4))

    def test_from_flat_list(self):
        m = matrix([1, 2, 3])
        self.assertEqual(m.rows, 3)
        self.assertEqual(m.cols, 1)
        self.assertEqual(m[0], mp.mpf(1))
        self.assertEqual(m[2], mp.mpf(3))

    def test_from_dimensions(self):
        m = matrix(3, 4)
        self.assertEqual(m.rows, 3)
        self.assertEqual(m.cols, 4)
        self.assertEqual(m[0, 0], mp.zero)
        self.assertEqual(m[2, 3], mp.zero)

    def test_from_square_dimension(self):
        m = matrix(3)
        self.assertEqual(m.rows, 3)
        self.assertEqual(m.cols, 3)

    def test_from_empty_list(self):
        m = matrix([])
        self.assertEqual(m.rows, 0)
        self.assertEqual(m.cols, 0)

    def test_from_mp_matrix(self):
        original = mp.matrix([[1, 2], [3, 4]])
        m = matrix(original)
        self.assertEqual(m.rows, 2)
        self.assertEqual(m.cols, 2)
        self.assertEqual(m[0, 0], mp.mpf(1))

    def test_from_matrix(self):
        original = matrix([[1, 2], [3, 4]])
        m = matrix(original)
        self.assertEqual(m[0, 0], mp.mpf(1))
        # Verify it's a deep copy
        original[0, 0] = mp.mpf(99)
        self.assertEqual(m[0, 0], mp.mpf(1))

    def test_invalid_type_raises(self):
        with self.assertRaises(TypeError):
            matrix("invalid")


class TestMatrixIndexing(unittest.TestCase):
    """Tests for matrix indexing (get/set)."""

    def test_getitem_2d(self):
        m = matrix([[1, 2], [3, 4]])
        self.assertEqual(m[0, 0], mp.mpf(1))
        self.assertEqual(m[0, 1], mp.mpf(2))
        self.assertEqual(m[1, 0], mp.mpf(3))
        self.assertEqual(m[1, 1], mp.mpf(4))

    def test_getitem_column_vector(self):
        m = matrix([10, 20, 30])
        self.assertEqual(m[0], mp.mpf(10))
        self.assertEqual(m[1], mp.mpf(20))
        self.assertEqual(m[2], mp.mpf(30))

    def test_getitem_row_vector(self):
        m = matrix([[10, 20, 30]])
        self.assertEqual(m[0], mp.mpf(10))
        self.assertEqual(m[2], mp.mpf(30))

    def test_setitem_2d(self):
        m = matrix(2, 2)
        m[0, 0] = mp.mpf(5)
        m[1, 1] = mp.mpf(7)
        self.assertEqual(m[0, 0], mp.mpf(5))
        self.assertEqual(m[1, 1], mp.mpf(7))

    def test_setitem_vector(self):
        m = matrix([0, 0, 0])
        m[1] = mp.mpf(42)
        self.assertEqual(m[1], mp.mpf(42))

    def test_invalid_index_raises(self):
        m = matrix([[1, 2], [3, 4]])
        with self.assertRaises(TypeError):
            _ = m[0]  # ambiguous for non-vector


class TestMatrixArithmetic(unittest.TestCase):
    """Tests for arithmetic operations."""

    def test_addition(self):
        a = matrix([[1, 2], [3, 4]])
        b = matrix([[5, 6], [7, 8]])
        c = a + b
        self.assertEqual(c[0, 0], mp.mpf(6))
        self.assertEqual(c[1, 1], mp.mpf(12))

    def test_subtraction(self):
        a = matrix([[5, 6], [7, 8]])
        b = matrix([[1, 2], [3, 4]])
        c = a - b
        self.assertEqual(c[0, 0], mp.mpf(4))
        self.assertEqual(c[1, 1], mp.mpf(4))

    def test_scalar_multiplication(self):
        a = matrix([[1, 2], [3, 4]])
        c = a * mp.mpf(2)
        self.assertEqual(c[0, 0], mp.mpf(2))
        self.assertEqual(c[1, 1], mp.mpf(8))

    def test_scalar_rmul(self):
        a = matrix([[1, 2], [3, 4]])
        c = mp.mpf(3) * a
        self.assertEqual(c[0, 0], mp.mpf(3))
        self.assertEqual(c[1, 1], mp.mpf(12))

    def test_matrix_multiplication(self):
        a = matrix([[1, 2], [3, 4]])
        b = matrix([[5, 6], [7, 8]])
        c = a * b
        # [1*5+2*7, 1*6+2*8] = [19, 22]
        # [3*5+4*7, 3*6+4*8] = [43, 50]
        self.assertEqual(c[0, 0], mp.mpf(19))
        self.assertEqual(c[0, 1], mp.mpf(22))
        self.assertEqual(c[1, 0], mp.mpf(43))
        self.assertEqual(c[1, 1], mp.mpf(50))

    def test_matrix_vector_multiplication(self):
        a = matrix([[1, 2], [3, 4]])
        v = matrix([1, 1])
        result = a * v
        self.assertEqual(result[0], mp.mpf(3))
        self.assertEqual(result[1], mp.mpf(7))

    def test_negation(self):
        a = matrix([[1, -2], [-3, 4]])
        b = -a
        self.assertEqual(b[0, 0], mp.mpf(-1))
        self.assertEqual(b[0, 1], mp.mpf(2))

    def test_division(self):
        a = matrix([[2, 4], [6, 8]])
        b = a / mp.mpf(2)
        self.assertEqual(b[0, 0], mp.mpf(1))
        self.assertEqual(b[1, 1], mp.mpf(4))

    def test_dimension_mismatch_add_raises(self):
        a = matrix([[1, 2], [3, 4]])
        b = matrix([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            _ = a + b

    def test_dimension_mismatch_mul_raises(self):
        a = matrix([[1, 2], [3, 4]])
        b = matrix([[1, 2, 3]])
        with self.assertRaises(ValueError):
            _ = a * b


class TestMatrixOperations(unittest.TestCase):
    """Tests for transpose, copy, len, etc."""

    def test_transpose(self):
        a = matrix([[1, 2, 3], [4, 5, 6]])
        b = a.T
        self.assertEqual(b.rows, 3)
        self.assertEqual(b.cols, 2)
        self.assertEqual(b[0, 0], mp.mpf(1))
        self.assertEqual(b[2, 1], mp.mpf(6))

    def test_copy_is_deep(self):
        a = matrix([[1, 2], [3, 4]])
        b = a.copy()
        b[0, 0] = mp.mpf(99)
        self.assertEqual(a[0, 0], mp.mpf(1))

    def test_len_column_vector(self):
        a = matrix([1, 2, 3])
        self.assertEqual(len(a), 3)

    def test_len_row_vector(self):
        a = matrix([[1, 2, 3]])
        self.assertEqual(len(a), 3)

    def test_len_matrix(self):
        a = matrix([[1, 2], [3, 4], [5, 6]])
        self.assertEqual(len(a), 3)  # number of rows

    def test_tolist(self):
        a = matrix([[1, 2], [3, 4]])
        data = a.tolist()
        self.assertEqual(len(data), 2)
        self.assertEqual(len(data[0]), 2)

    def test_swap_row(self):
        a = matrix([[1, 2], [3, 4]])
        a.swap_row(0, 1)
        self.assertEqual(a[0, 0], mp.mpf(3))
        self.assertEqual(a[1, 0], mp.mpf(1))

    def test_str(self):
        a = matrix([[1, 2], [3, 4]])
        s = str(a)
        self.assertIn("1", s)
        self.assertIn("4", s)


class TestMatrixEdgeCases(unittest.TestCase):
    """Edge case tests."""

    def test_1x1_matrix(self):
        a = matrix([[5]])
        self.assertEqual(a.rows, 1)
        self.assertEqual(a.cols, 1)
        self.assertEqual(a[0, 0], mp.mpf(5))

    def test_identity_multiply(self):
        a = matrix([[1, 0], [0, 1]])
        b = matrix([[3, 7], [2, 5]])
        c = a * b
        self.assertEqual(c[0, 0], mp.mpf(3))
        self.assertEqual(c[1, 1], mp.mpf(5))

    def test_rsub(self):
        a = matrix([[1, 2], [3, 4]])
        b = matrix([[10, 10], [10, 10]])
        c = b - a  # uses __sub__
        d = a.__rsub__(b)  # -a + b
        self.assertEqual(c[0, 0], d[0, 0])
        self.assertEqual(c[1, 1], d[1, 1])


if __name__ == '__main__':
    unittest.main()
