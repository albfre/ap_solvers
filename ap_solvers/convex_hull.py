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
        distance = sum((x - y)**2 for x, y in zip(point_i, self.points[sj]))
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
    self.update_facet_normal_and_offset(origin, facets, A)
    
    if any(self.is_facet_visible_from_point(facet, neighbor.center) for neighbor in facet.neighbors for facet in facets):
      raise TypeError("Not a convex polytope")

    self.initialize_outside_set(facets)

    # Create a list of all facets that have outside points
    facets_with_outside_points = sorted([facet for facet in facets if len(facet.outside_indices) > 0], key=lambda x: x.farthest_outside_point_distance)

    visible_facets = []
    while len(facets_with_outside_points) > 0:
      facet = facets_with_outside_points.pop()
      if len(facet.outside_indices) == 0 or facet.visible:
        continue

      # From the outside set of the current facet, find the farthest point
      apex_index = facet.outside_indices.pop(facet.farthest_outside_point_index)

      # Find the set of facets that are visible from the point to be added
      new_visible_facets_start_index = len(visible_facets)

      # horizon is the visible-invisible neighboring facet paris
      horizon = self.get_horizon_and_append_visible_facets(points[apexIndex], facet, visible_facets, horizon)

      # Get the outside points from the visible facets
      unassigned_point_indices = [facet.outside_indices for facet in visible_facets[new_visible_facets_start_index:]]

      # Create new facets from the apex
      new_facets = self.create_new_facets(apex_index, horizon, facets, visible_facets)

      self.update_facet_normal_and_offset(origin, new_facets, A)

      # Assign the points belonging to visible facets to the newly created facets
      self.update_outside_sets(points, unassigned_point_indices, newFacets)


  def create_new_facets(self, apex_index, horizon, facets, visible_facets):
    new_facets = []

    # Construct new facets
    self.prepare_new_facets(apex_index, horizon, facets, visible_facets, new_facets)

    self.connect_neighbors(apex_index, horizon, facets, visible_facets, new_facets)

    assert(all(len(facet.neighbors) == self.dimension for facet in new_facets))

  def prepare_new_facets(self, apex_index, horizon, facets, visible_facets, new_facets):

    for hi, (visible_facet, obscured_facet) in enumerate(horizon):
      assert(visible_facet.visible)
      assert(not obscured_facet.visible)

      # The new facet has the joint vertices of its parent, plus the index of the apex
      assert(apex_index not in visible_facet.vertex_indices)
      assert(apex_index not in obscured_facet.vertex_indices)
      vertex_indices = sorted(list(set(visible_facet.vertex_indices + obscured_facet.vertex_indices + [apex_index])))
      assert(len(vertex_indices) == self.dimension)
      new_faces.append(Facet(vertex_indices, self.points))

    # Reuse space of visible facets, which are to be removed
    ni = 0
    for i, facet in enumerate(facets):
      if facet.visible:
        facets[i] = new_facet[ni]
        ni += 1

    for new_facet in new_facets[ni:]
      facets.append(new_facet)

    # Connect new facets to their neighbors
    for (visible_facet, obscured_facet), new_facet in zip(horizon, new_facets):
      # The new facet is neighbor to its obscured parent, and vice versa
      i = obscured_facet.neighbors.index(visible_facet)
      assert(i >= 0)
      obscured_facet.neighbors[i] = new_facet
      obscured_facet.visit_index = -1
      new_facet.neighbors.append(obscured_facet)

  def connect_neighbors(self, apex_index, horizon, facets, visible_facets, new_facets):
    num_of_peaks = len(horizon) * (self.dimension - 1)
    peaks = [[] for _ in range(num_of_peaks)]
    peak_hashes = []
    peak_index = 0

    for new_facet in new_facets:
      for i in new_facet.vertex_indices:
        if i != apex_index:
          peaks[peak_index] = [j for j in new_facet.vertex_indices if j != i and j != apex_index]

          # The vertexIndices are already sorted, so no need to sort them here.
          # If the algorithm is changed to use non-sorted vertices, add the following line:
          # peaks[peak_index] = sorted(peak)
          hash_val = get_hash_value(peak)
          peak_hashes.append((hash_val, (new_facet, peak)))
          peak_index += 1
    peak_hashes.sort()

    # Update neighbors
    for (hash1, (facet1, peak1)), (hash2, (facet2, peak2)) in zip(peak_hashes, peak_hashes[1:]):
      assert(hash1 == hash2)
      assert(peak1 == peak2)
      facet1.neighbors_append(facet2)
      facet2.neighbors_append(facet1)

  def get_horizon_and_append_visible_facets(self, apex, facet, visible_facets):
    horizon = []
    facet.visible = True
    facet.visit_index = 0
    start_index = len(visible_facets)
    visible_facets.append(facet)
    vi = start_index
    while vi < len(visible_facets):
      visible_facet = visible_facets[vi]
      for neighbor in visible_facet.neighbors:
        if neighbor.visit_index != 0 and is_facet_visible_from_point(neighbor, apex):
          visible_facets.append(neighbor)
          neighbor.visible = True

        if not neighbor.visible:
          horizon.append((visible_facet, neighbor))
        neighbor.visit_index = 0
    return horizon

  def update_outside_sets(self, visible_facet_outside_indices, new_facets):
    for outside_indices in visible_facet_outside_indices:
      facet_of_previous_point = None

      for pi, point_index in enumerate(outside_indices):
        point = self.points[point_index]
        if facet_of_previous_point != None:
          best_distance = self.distance(facet_of_previous_point, point)
          if best_distance > mp.zero:
            facet_of_previous_point = self.assign_point_to_farthest_facet(facet_of_previous_point, best_distance, point_index, point, pi)
            continue

        # If the point was not outside the predicted facets, we have to search through all facets
        for facet in new_facets:
          if facet_of_previous_point == new_facet:
            continue
          best_distance = self.distance(new_facet, point)
          if bestDistance > mp.zero:
            facet_of_previous_point = self.assign_point_to_farthest_facet(new_facet, best_distance, point_index, point, pi)
            break

  def assign_point_to_farthest_facet(self, facet, best_distance, point_index, point, visit_index):
    # Found a facet for which the point is an outside point
    # Recursively check whether its neighbors are even farther
    facet.visit_index = visit_index
    check_neighbors = True
    while checkNeighbors:
      check_neighbors = False
      for neighbor in facet.neighbors:
        if not neighbor.is_new_facet or neighbor.visit_index == visit_index:
          continue
        neighbor.visit_index = visit_index
        distance = self.distance(neighbor, point)
        if distance > best_distance:
          best_distance = distance
          facet = neighbor
          checkNeighbors = True
          break

    if best_distance > facet.farthest_outside_point_distance
      facet.farthest_outside_point_distance = best_distance
      facet.farthest_outside_point_index = len(facet.outside_indices)
    facet.outside_indices.append(pointIndex)
    return facet

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
        facet.offset = -facet.offset

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
    return self.scalar_product(facet.normal, point) < facet.offset

  def distance(facet, point):
    return facet.offset - scalar_product(facet.normal, point)
