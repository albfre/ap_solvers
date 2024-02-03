from ap_solvers import convex_hull
from scipy.spatial import ConvexHull

from mpmath import mp
import time
import random

import unittest
from parameterized import parameterized

class TestConvexHull(unittest.TestCase):
  def compare_to_scipy(self, points):
    print("Compute convex hull with solver")
    tic = time.time()
    ch = convex_hull.ConvexHull(points)
    toc = time.time() - tic
    print("Time: " + str(toc))
    print("Num vertices: %s" % len(ch.vertices))
    print("Num simplices: %s" % len(ch.simplices))
    simplices1 = sorted([sorted(simplex) for simplex in ch.simplices])
    vertices1 = sorted(ch.vertices)

    print("Compute convex hull with scipy")
    tic = time.time()
    ch2 = ConvexHull(points)
    toc = time.time() - tic
    print("Time: " + str(toc))
    simplices2 = sorted([sorted(simplex) for simplex in ch2.simplices])
    vertices2 = sorted(ch2.vertices)
    self.assertTrue(vertices1 == vertices2)
    self.assertTrue(simplices1 == simplices2)
    for eq1 in ch.equations:
      min_diff1 = min(sum(abs(x1 - x2) for x1, x2 in zip(eq1, eq2)) for eq2 in ch2.equations)
      min_diff2 = min(sum(abs(x1 + x2) for x1, x2 in zip(eq1, eq2)) for eq2 in ch2.equations)
      min_diff = min(min_diff1, min_diff2) 
      self.assertTrue(min_diff < 1e-8)

  def test_simple_3D(self):
    print("Test simple shape in 3D")
    mp.dps = 100
    points = [[0, 0, 0],
              [0, 0, 1], 
              [0, 1, 0], 
              [1, 0, 0], 
              [0.2, 0.2, 0.2], 
              [0.7, 0.7, 0.7], 
              [0.3, 0.3, 0.3]]
    self.compare_to_scipy(points)

  @parameterized.expand(range(2, 5))
  def test_dimensions(self, dimension):
    print("Computing convex hull in dimension %s" % dimension)
    mp.dps = 50
    num_points = 20
    points = []
    random.seed(17)
    points = [[random.random() for _ in range(dimension)] for _ in range(num_points)]
    self.compare_to_scipy(points)

if __name__ == "__main__":
  unittest.main()

