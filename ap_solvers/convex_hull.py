from ap_solvers import dense_mp_matrix
from mpmath import mp

class Facet:
  def __init__(self, vertex_indices, points):
    self.vertex_indices = vertex_indices
    self.visit_index = -1
    self.farthest_outside_point_distance = mp.zero
    self.farthest_outside_point_index = -1
    self.visible = False
    self.is_new_facet = True
    self.has_obscured = False
    self.neighbors = []
    self.offset = mp.zero
    self.center = [sum(points[vi][d] for vi in self.vertex_indices) / self.dimension for d in range(dimension)]
    
class ConvexHull:
  def __init__(self, points)
    self.original_points = [tuple(p) for p in points]
    self.points = list(set(self.original_points))
    num_points = len(points)
    self.dimension = 0 if num_points == 0 else len(points[0])

    if num_points <= self.dimension:
      raise TypeError("Too few points (%s) to compute hull in dimension %s" % (num_points, self.dimension)

    if any(len(p) != self.dimension for p in self.points):
      raise TypeError("All points must have the correct dimension")

    self.vertices = self.compute()

  def compute(self):
    self.facets = self.initial_simplex()
    self.grow_convex_hull()
    vertex_indices = []
    for facet in self.facets:
      vertex_indices.append(facet.vertex_indices)
      vis = vertex_indices[-1]
      for i in range(dimension):
        vis[i] = self.original_points.index(self.points[vis[i]])

    return vertex_indices

  def initial_indices(self):
    indices = set()
    for d in range(self.dimension):
      min_element = min(self.points, key=lambda p: p[d])
      max_element = max(self.points, key=lambda p: p[d])
      indices.add(points.index(min_element))
      indices.add(points.index(max_element))
    i = 0
    while len(indices) <= dimension:
      indices.add(i)
      i += 1

    return list(indices)
    
    
  def initial_simplex(self):
    initial_indices = self.initial_indices()
    facets = []

    # Compute pairwise squared distances
    distances = []
    for i, si in enumerate(initial_indices):
      point_i = self.points[si]
      for sj in initial_indices[i + 1:]:
        point_j = points[sj]
        distance = sum((x - y)**2 for x, y in zip(point_i, point_j))
        distances.append([distance, (si, sj)])
    distances = sorted(distances)

    # Select the indices of pairs with large distance
    indices = set()
    while len(indices) <= dimension:
      index_pair = distances.pop()[1]
      indices.add(index_pair[0])
      indices.add(index_pair[1])
    indices = list(indices)

    # Create initial simplex using the (dimension + 1) first points.
    # The facets have vertices [0, ..., dimension - 1], [1, ..., dimension], ..., [dimension, 0, ..., dimension - 2] in sortedIndices.
    for i in range(self.dimension + 1):
      vertex_indices = [0] * dimension
      for j in range(self.dimension):
        vertex_indices[j] = indices[(i + j) % (dimension + 1)]
      facest.append(Facet(vertex_indices, self.points))
      facets[-1].is_new_facet = False

    # Update the facets' neighbors

    for facet in facets:
      facet.neighbors = [f for f in facets if f != facet]
    return facets

  def grow_convex_hull(self):
    facets = self.facets
    if any(len(facet.vertex_indices) != dimension for facet in facets):
      raise TypeError("All facets must be full dimensional")

    # The vertex indices are assumed to be sorted when new facets are created
    for facet in facets:
      facet.vertex_indices = sorted(facet.vertex_indices)

    # Compute origin as the mean of the center points of the seed facets
    origin = [mp.zero] * self.dimension
    for f in facets:
      origin = [x + y for x, y in zip(origin, f.center)]
    origin = [x / len(facets) for x in origin]

    # Compute inwards-oriented facet normals
    A = dense_mp_matrix.matrix(dimension, dimension)
    self.update_facet_normal_and_offset(origin, facets, A);
    
    if any(self.is_facet_visible_from_point(facet, neighbor.center) for neighbor in facet.neighbors for facet in facets):
      raise TypeError("Not a convex polytope")

    self.initialize_outside_set(facets)

    # Create a list of all facets that have outside points
    facets_with_outside_points = [facet for facet in facets if len(facet.outside_indices) > 0]

  def update_facet_normal_and_offset(self, origin, facets, A):
    assert(A.rows == self.dimension)
    assert(A.cols == self.dimension)
    for facet in facets:
      assert(len(facet.vertex_indices) == dimension)
      p1 = self.points[facet.vertex_indices[0]]
      for i in range(dimension - 1):
        for j in range(dimension):
          A[i, j] = points[facet.vertex_indicex[i + 1]][j] - p1[j]
      b = [mp.zero] * self.dimension
      b[-1] = mp.one
      A[dimension - 1, dimension - 1] = mp.one

      # Solve A x = b
      overwritingSolveLinearSystemOfEquations_(A, b)
      abs_sum = sum(abs(bi) for bi in b)
      b = [bi / abs_sum for bi in b]

      facet.offset = self.scalar_product(facet.normal, self.points[facet.vertex_indices[0]])
      self.hyper_planes += 1

      # Orient normal inwards
      if is_facet_visible_from_point(facet, origin):
        facet.normal = [-n for n in facet.normal]
        facet.offset = -facet.offset;

  def initialize_outside_set(self, facets):
    check_point = [True] * len(self.points)
    for facet in facets:
      for i in facet.vertex_indices:
        check_point[i] = False

    for pi, point in enumerate(self.points):
      if check_point[pi]:
        max_distance = mp.zero
        farthest_facet = None
        for facet in facets:
          distance = self.distance(facet, point)
          if distance > max_distance:
            max_distance = distance
            farthest_facet = facet
      if farthest_facet:
        if max_distance > farthest_facet.farthest_outside_point_distance:
          farthest_facet.farthest_outside_point_distance = max_distance
          farthest_facet.farthest_outside_point_index = len(farthest_facet.outside_indices)
        farthest_facet.outside_indices.append(pi)

  def scalar_product(self, a, b):
    self.disstance_tests += 1
    return sum(x * y for x, y in zip(a, b))

  def is_facet_visible_from_point(facet, point):
    # Returns true if the point is contained in the open negative halfspace of the facet
    return self.scalar_product(facet.normal, point) < facet.offset;

  def distance(facet, point):
    return facet.offset - scalar_product(facet.normal, point);
