from ap_solvers import nelder_mead

from mpmath import mp
import numpy as np
import time

import unittest
from parameterized import parameterized


class TestNelderMead(unittest.TestCase):
    def test_simple_quadratic(self):
        """Test optimization of a simple quadratic function f(x,y) = x^2 + y^2"""
        mp.dps = 50
        
        def f(x):
            return x[0]**2 + x[1]**2
        
        x_start = [mp.mpf('1.0'), mp.mpf('1.0')]
        x, f_val, status = nelder_mead.nelder_mead(f, x_start, max_iter=1000, no_improve_thr=mp.mpf('1e-40'))
        
        # Should converge to (0, 0) with value 0
        self.assertTrue(abs(x[0]) < mp.mpf('1e-10'))
        self.assertTrue(abs(x[1]) < mp.mpf('1e-10'))
        self.assertTrue(abs(f_val) < mp.mpf('1e-20'))
        self.assertIn(status, ["Optimal solution found", "Maximum number of iterations"])

    def test_rosenbrock(self):
        """Test optimization of Rosenbrock function"""
        mp.dps = 50
        
        def rosenbrock(x):
            """Rosenbrock function: (1-x)^2 + 100(y-x^2)^2, minimum at (1,1)"""
            return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
        
        x_start = [mp.mpf('0.0'), mp.mpf('0.0')]
        x, f_val, status = nelder_mead.nelder_mead(rosenbrock, x_start, max_iter=5000, no_improve_thr=mp.mpf('1e-40'))
        
        # Should converge close to (1, 1)
        self.assertTrue(abs(x[0] - 1) < mp.mpf('1e-5'))
        self.assertTrue(abs(x[1] - 1) < mp.mpf('1e-5'))
        self.assertTrue(abs(f_val) < mp.mpf('1e-8'))

    def test_sphere_3d(self):
        """Test optimization of 3D sphere function f(x,y,z) = x^2 + y^2 + z^2"""
        mp.dps = 50
        
        def sphere(x):
            return sum([xi**2 for xi in x])
        
        x_start = [mp.mpf('2.0'), mp.mpf('3.0'), mp.mpf('-1.0')]
        x, f_val, status = nelder_mead.nelder_mead(sphere, x_start, max_iter=1000, no_improve_thr=mp.mpf('1e-40'))
        
        # Should converge to (0, 0, 0) with value 0
        for xi in x:
            self.assertTrue(abs(xi) < mp.mpf('1e-10'))
        self.assertTrue(abs(f_val) < mp.mpf('1e-20'))

    def test_booth_function(self):
        """Test optimization of Booth function: (x + 2y - 7)^2 + (2x + y - 5)^2"""
        mp.dps = 50
        
        def booth(x):
            return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2
        
        x_start = [mp.mpf('0.0'), mp.mpf('0.0')]
        x, f_val, status = nelder_mead.nelder_mead(booth, x_start, max_iter=1000, no_improve_thr=mp.mpf('1e-40'))
        
        # Minimum at (1, 3) with value 0
        self.assertTrue(abs(x[0] - 1) < mp.mpf('1e-8'))
        self.assertTrue(abs(x[1] - 3) < mp.mpf('1e-8'))
        self.assertTrue(abs(f_val) < mp.mpf('1e-15'))

    def test_beale_function(self):
        """Test optimization of Beale function"""
        mp.dps = 50
        
        def beale(x):
            """Beale function, minimum at (3, 0.5) with value 0"""
            return ((mp.mpf('1.5') - x[0] + x[0]*x[1])**2 + 
                    (mp.mpf('2.25') - x[0] + x[0]*x[1]**2)**2 + 
                    (mp.mpf('2.625') - x[0] + x[0]*x[1]**3)**2)
        
        x_start = [mp.mpf('1.0'), mp.mpf('1.0')]
        x, f_val, status = nelder_mead.nelder_mead(beale, x_start, max_iter=5000, no_improve_thr=mp.mpf('1e-40'))
        
        # Should converge close to (3, 0.5)
        self.assertTrue(abs(x[0] - 3) < mp.mpf('1e-5'))
        self.assertTrue(abs(x[1] - 0.5) < mp.mpf('1e-5'))
        self.assertTrue(abs(f_val) < mp.mpf('1e-8'))

    @parameterized.expand([[50], [100], [150]])
    def test_quadratic_different_precision(self, dps):
        """Test with different precision levels"""
        mp.dps = dps
        
        def f(x):
            return (x[0] - 2)**2 + (x[1] + 3)**2
        
        x_start = [mp.mpf('0.0'), mp.mpf('0.0')]
        x, f_val, status = nelder_mead.nelder_mead(f, x_start, max_iter=1000, 
                                        no_improve_thr=mp.mpf('1e-' + str(dps-10)))
        
        # Should converge to (2, -3) with value 0
        self.assertTrue(abs(x[0] - 2) < mp.mpf('1e-8'))
        self.assertTrue(abs(x[1] + 3) < mp.mpf('1e-8'))
        self.assertTrue(abs(f_val) < mp.mpf('1e-15'))

    def test_convergence_with_parameters(self):
        """Test that algorithm parameters affect convergence"""
        mp.dps = 50
        
        def f(x):
            return x[0]**2 + x[1]**2
        
        x_start = [mp.mpf('5.0'), mp.mpf('5.0')]
        
        # Test with different alpha values
        x1, f_val1, status1 = nelder_mead.nelder_mead(f, x_start, alpha=1.0, max_iter=500)
        x2, f_val2, status2 = nelder_mead.nelder_mead(f, x_start, alpha=1.5, max_iter=500)
        
        # Both should converge to minimum
        self.assertTrue(abs(f_val1) < mp.mpf('1e-10'))
        self.assertTrue(abs(f_val2) < mp.mpf('1e-10'))


if __name__ == "__main__":
    unittest.main()
