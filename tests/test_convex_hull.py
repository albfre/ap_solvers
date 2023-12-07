from ap_solvers import convex_hull
from ap_solvers import dense_mp_matrix

from mpmath import mp
import time
import random

import unittest
from parameterized import parameterized

class TestConvexHull(unittest.TestCase):
  @parameterized.expand(range(1, 5))
  def test_dimensions(self, dimension):
    print("Computing convex hull in dimension %s" % dimension)
    mp.dps = 50
    num_points = 20
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
              [0, 1, 1], 
              [1, 0, 0], 
              [1, 0, 1], 
              [1, 1, 0], 
              [1, 1, 1]]

    random.seed(17)
    magnitude = 0
    for p in points:
      for i in range(len(p)):
        p[i] += random.uniform(-magnitude, magnitude)
    
    print(points)

    # Compute the convex hull
    tic = time.time()
    ch = convex_hull.ConvexHull(points)
    toc = time.time() - tic
    print("Time: " + str(toc))
    print(len(ch.vertices))
    print(ch.vertices)

if __name__ == "__main__":
  unittest.main()

