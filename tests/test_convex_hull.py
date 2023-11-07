from ap_solvers import convex_hull
from ap_solvers import dense_mp_matrix

from mpmath import mp
import scipy
import numpy as np
import time
import random

import unittest
from parameterized import parameterized

class TestConvexHull(unittest.TestCase):
  def test_convex_hull(self):
    points = [[0, 0, 0],
              [0, 0, 1], 
              [0, 1, 0], 
              [0, 1, 1], 
              [1, 0, 0], 
              [1, 0, 1], 
              [1, 1, 0], 
              [1, 1, 1]]

    points = [[0,0],
              [0,1],
              [1,0],
              [0.9, 0.9]]
    random.seed(17)
    magnitude = 1e-7

    for p in points:
      for i in range(len(p)):
        p[i] += random.uniform(-magnitude, magnitude)
    
    print(points)

    # Compute result using bunch_kaufman implementation
    tic = time.time()
    ch = convex_hull.ConvexHull(points)
    toc = time.time() - tic
    print("Time: " + str(toc))
    print(len(ch.vertices))
    print(ch.vertices)

if __name__ == "__main__":
  unittest.main()

