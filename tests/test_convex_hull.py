from ap_solvers import convex_hull
from scipy.spatial import ConvexHull

from mpmath import mp
import time
import random

import unittest
from parameterized import parameterized

class TestConvexHull(unittest.TestCase):
  @parameterized.expand(range(1, 5))
  def _test_dimensions(self, dimension):
    print("Computing convex hull in dimension %s" % dimension)
    mp.dps = 50
    num_points = 100
    points = []
    random.seed(17)
    points = [[random.random() for _ in range(dimension)] for _ in range(num_points)]
    tic = time.time()
    ch = convex_hull.ConvexHull(points)
    toc = time.time() - tic
    print("Time: " + str(toc))
    print("Num vertices: %s" % len(ch.vertices))

  def test_convex_hull(self):
    mp.dps = 100
    points = [[0, 0, 0],
              [0, 0, 1], 
              [0, 1, 0], 
              [1, 0, 0], 
              [0.2, 0.2, 0.2], 
              [0.7, 0.7, 0.7], 
              [0.3, 0.3, 0.3]]

    # Compute the convex hull
    tic = time.time()
    ch = convex_hull.ConvexHull(points)
    toc = time.time() - tic
    print("Time: " + str(toc))
    print("Num vertices: %s" % len(ch.vertices))
    print(ch.vertices)
    print("Num num simplices: %s" % len(ch.simplices))
    simplices1 = sorted([sorted(simplex) for simplex in ch.simplices])
    print(simplices1)
    print(ch.equations)

    ch2 = ConvexHull(points)
    simplices2 = sorted([sorted(simplex) for simplex in ch2.simplices])
    print("Num vertices: %s" % len(ch2.vertices))
    print(ch2.vertices)
    print("Num num simplices: %s" % len(ch.simplices))
    self.assertTrue(all(ch.vertices == ch2.vertices))
    self.assertTrue(simplices1 == simplices2)
    print(ch2.equations)


if __name__ == "__main__":
  unittest.main()

