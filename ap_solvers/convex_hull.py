from ap_solvers import dense_mp_matrix
from mpmath import mp
import random
from itertools import starmap
from operator import mul

class Facet:
  def __init__(self, vertex_indices, points):
    self.vertex_indices = vertex_indices
    self.visit_index = -1
    self.outside_indices = []
    self.farthest_outside_point_distance = mp.zero
    self.farthest_outside_point_index = -1
    self.visible = False
    self.is_new_facet = True
    self.has_obscured = False
    self.neighbors = []
    self.offset = mp.zero
    dimension = len(points[0])
    self.normal = [mp.zero] * dimension
    self.center = [sum(points[vi][d] for vi in self.vertex_indices) / dimension for d in range(dimension)]
    
class ConvexHull:
  def __init__(self, points):
    """
    Initialize the ConvexHull object with a set of points.

    Args:
        points (list of tuples/lists): List of points in a high-dimensional space.
                                 Each point should be represented as a tuple or list of equal length.
    Raises:
        ValueError: If the input list of points is empty or there are too few points.
        TypeError: If the points are of erroneus dimension.
    """
    if not points:
      raise ValueError("Points must not be empty")
    self.original_points = [tuple(mp.mpf(x) for x in p) for p in points]
    self.unperturbed_points = list(set(self.original_points))
    self.points = self.unperturbed_points
    num_points = len(points)
    self.dimension = len(points[0])
    self.powers_of_twelve = list(reversed([i ** 12 for i in range(1, self.dimension - 1)]))
    self.distance_tests = 0
    self.hyper_planes = 0
    self.A = dense_mp_matrix.matrix(self.dimension, self.dimension)

    self.max_perturbation_iter = 5
    self.perturbation_iter = 0
    self.shortest_distance = mp.zero
    random.seed(17)

    if num_points <= self.dimension:
      raise ValueError("Too few points (%s) to compute hull in dimension %s" % (num_points, self.dimension))

    if any(len(p) != self.dimension for p in self.points):
      raise TypeError("All points must have the correct dimension")

    self.vertices = self.compute()

  def compute(self):
    self.facets = self.initial_simplex()
    try:
      self.grow_convex_hull()
    except Exception as e:
      print("An exception occurred:", str(e))
      if self.perturbation_iter == 0:
        self.shortest_distance = min(self.point_distance(p1, p2)
            for i, p1 in enumerate(self.unperturbed_points)
            for j, p2 in enumerate(self.unperturbed_points)
            if i < j)
      self.perturbation_iter += 1
      if self.perturbation_iter > self.max_perturbation_iter:
        raise ValueError("Unable to compute convex hull")
      if 0.5 * self.shortest_distance > mp.dps:
        raise ValueError("Shortest distance between points is too small in comparison with mp.dps")
      factor = next(x for x in range(20, -1, -1) if x * self.max_perturbation_iter < mp.dps)
      perturbation = 0.5 * self.shortest_distance * 10 ** -(factor * (self.max_perturbation_iter - self.perturbation_iter))
      print("Perturbing with size: %s" % perturbation)
      self.points = [tuple([coord + random.uniform(-perturbation, perturbation) for coord in p]) for p in self.unperturbed_points]
      return self.compute()

    vertex_indices = []
    for facet in self.facets:
      vertex_indices.append(facet.vertex_indices)
      vis = vertex_indices[-1]
      for i in range(self.dimension):
        vis[i] = self.original_points.index(self.unperturbed_points[vis[i]])

    return vertex_indices

  def initial_indices(self):
    indices = set()
    for d in range(self.dimension):
      min_element = min(self.points, key=lambda p: p[d])
      max_element = max(self.points, key=lambda p: p[d])
      indices.add(self.points.index(min_element))
      indices.add(self.points.index(max_element))
    i = 0
    while len(indices) <= self.dimension:
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
    while len(indices) <= self.dimension:
      index_pair = distances.pop()[1]
      indices.add(index_pair[0])
      indices.add(index_pair[1])
    indices = list(indices)

    # Create initial simplex using the (dimension + 1) first points.
    # The facets have vertices [0, ..., dimension - 1], [1, ..., dimension], ..., [dimension, 0, ..., dimension - 2] in sortedIndices.
    for i in range(self.dimension + 1):
      vertex_indices = [0] * self.dimension
      for j in range(self.dimension):
        vertex_indices[j] = indices[(i + j) % (self.dimension + 1)]
      facets.append(Facet(vertex_indices, self.points))
      facets[-1].is_new_facet = False

    # Update the facets' neighbors
    for facet in facets:
      facet.neighbors = [f for f in facets if f != facet]
    return facets

  def grow_convex_hull(self):
    facets = self.facets
    if any(len(facet.vertex_indices) != self.dimension for facet in facets):
      raise TypeError("All facets must be full dimensional")

    # The vertex indices are assumed to be sorted when new facets are created
    for facet in facets:
      facet.vertex_indices = sorted(facet.vertex_indices)

    # Compute origin as the mean of the center points of the seed facets
    origin = [mp.zero] * self.dimension
    for facet in facets:
      origin = [x + y for x, y in zip(origin, facet.center)]
    origin = [x / len(facets) for x in origin]

    # Compute inwards-oriented facet normals
    self.update_facet_normal_and_offset(origin, facets)
    
    self.throw_if_not_convex_polytope(facets)

    self.initialize_outside_set(facets)

    # Create a list of all facets that have outside points
    facets_with_outside_points = sorted([facet for facet in facets if len(facet.outside_indices) > 0], key=lambda x: x.farthest_outside_point_distance)

    visible_facets = []
    while facets_with_outside_points:
      facet = facets_with_outside_points.pop()
      if not facet.outside_indices or facet.visible:
        continue

      # From the outside set of the current facet, find the farthest point
      apex_index = facet.outside_indices.pop(facet.farthest_outside_point_index)

      # Find the set of facets that are visible from the point to be added
      new_visible_facets_start_index = len(visible_facets)

      # horizon is the visible-invisible neighboring facet paris
      horizon = self.get_horizon_and_append_visible_facets(self.points[apex_index], facet, visible_facets)

      # Get the outside points from the visible facets
      unassigned_point_indices = [facet.outside_indices for facet in visible_facets[new_visible_facets_start_index:]]

      # Create new facets from the apex
      new_facets = self.prepare_new_facets(apex_index, horizon, facets, visible_facets)
      self.connect_neighbors(apex_index, horizon, facets, new_facets)
      assert(all(len(facet.neighbors) == self.dimension for facet in new_facets))

      self.update_facet_normal_and_offset(origin, new_facets)

      # Assign the points belonging to visible facets to the newly created facets
      self.update_outside_sets(unassigned_point_indices, new_facets)

      # Add the new facets with outside points to the vector of all facets with outside points
      for facet in new_facets:
        facet.is_new_facet = False
        facet.visit_index = -1
      new_facets = sorted([facet for facet in new_facets if len(facet.outside_indices) > 0], key=lambda x: x.farthest_outside_point_distance)

      if new_facets and facets_with_outside_points and new_facets[-1].farthest_outside_point_distance > facets_with_outside_points[-1].farthest_outside_point_distance:
        facets_with_outside_points += new_facets
      else:
        facets_with_outside_points = new_facets + facets_with_outside_points

    facets = [facet for facet in facets if not facet.visible]

    self.throw_if_not_convex_polytope(facets)

  def prepare_new_facets(self, apex_index, horizon, facets, visible_facets):
    new_facets = []

    for hi, (visible_facet, obscured_facet) in enumerate(horizon):
      assert(visible_facet.visible)
      assert(not obscured_facet.visible)

      # The new facet has the joint vertices of its parent, plus the index of the apex
      assert(apex_index not in visible_facet.vertex_indices)
      assert(apex_index not in obscured_facet.vertex_indices)
      vertex_indices = sorted(list(set(visible_facet.vertex_indices) & set(obscured_facet.vertex_indices)) + [apex_index])
      assert len(vertex_indices) == self.dimension, "Vertex: %s, dimension: %s" % (len(vertex_indices), self.dimension)
      new_facets.append(Facet(vertex_indices, self.points))

    # Reuse space of visible facets, which are to be removed
    ni = 0
    for i, facet in enumerate(facets):
      if facet.visible:
        facets[i] = new_facets[ni]
        ni += 1
      if ni >= len(new_facets):
        break

    for new_facet in new_facets[ni:]:
      facets.append(new_facet)

    # Connect new facets to their neighbors
    for (visible_facet, obscured_facet), new_facet in zip(horizon, new_facets):
      # The new facet is neighbor to its obscured parent, and vice versa
      i = obscured_facet.neighbors.index(visible_facet)
      assert(i >= 0)
      obscured_facet.neighbors[i] = new_facet
      obscured_facet.visit_index = -1
      new_facet.neighbors.append(obscured_facet)

    return new_facets

  def connect_neighbors(self, apex_index, horizon, facets, new_facets):
    assert(len(new_facets) == len(horizon))
    num_of_peaks = len(horizon) * (self.dimension - 1)
    peaks = [[] for _ in range(num_of_peaks)]
    peak_hashes = []
    peak_index = 0

    for new_facet in new_facets:
      for i in new_facet.vertex_indices:
        if i != apex_index:
          peaks[peak_index] = [j for j in new_facet.vertex_indices if j != i and j != apex_index]

          # The vertexIndices are already sorted, so no need to sort them here.
          peak = peaks[peak_index]
          hash_val = sum(vi * power for vi, power in zip(peak, self.powers_of_twelve))
          peak_hashes.append((hash_val, peak, new_facet))
          peak_index += 1
    peak_hashes.sort(key = lambda x: (x[0], x[1]))

    # Update neighbors
    for (hash1, peak1, facet1), (hash2, peak2, facet2) in zip(peak_hashes[::2], peak_hashes[1::2]):
      assert(hash1 == hash2)
      assert(peak1 == peak2)
      facet1.neighbors.append(facet2)
      facet2.neighbors.append(facet1)

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
        if neighbor.visit_index != 0 and self.is_facet_visible_from_point(neighbor, apex):
          visible_facets.append(neighbor)
          neighbor.visible = True

        if not neighbor.visible:
          horizon.append((visible_facet, neighbor))
        neighbor.visit_index = 0
      vi += 1
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
        for new_facet in new_facets:
          if facet_of_previous_point == new_facet:
            continue
          best_distance = self.distance(new_facet, point)
          if best_distance > mp.zero:
            facet_of_previous_point = self.assign_point_to_farthest_facet(new_facet, best_distance, point_index, point, pi)
            break

  def assign_point_to_farthest_facet(self, facet, best_distance, point_index, point, visit_index):
    # Found a facet for which the point is an outside point
    # Recursively check whether its neighbors are even farther
    facet.visit_index = visit_index
    check_neighbors = True
    while check_neighbors:
      check_neighbors = False
      for neighbor in facet.neighbors:
        if not neighbor.is_new_facet or neighbor.visit_index == visit_index:
          continue
        neighbor.visit_index = visit_index
        distance = self.distance(neighbor, point)
        if distance > best_distance:
          best_distance = distance
          facet = neighbor
          check_neighbors = True
          break

    if best_distance > facet.farthest_outside_point_distance:
      facet.farthest_outside_point_distance = best_distance
      facet.farthest_outside_point_index = len(facet.outside_indices)
    facet.outside_indices.append(point_index)
    return facet

  def update_facet_normal_and_offset(self, origin, facets):
    dimension = self.dimension
    A = self.A
    assert(A.rows == self.dimension)
    assert(A.cols == self.dimension)
    for facet in facets:
      assert(len(facet.vertex_indices) == dimension)
      p1 = self.points[facet.vertex_indices[0]]
      for i in range(dimension - 1):
        data_i = A.data()[i]
        pi = self.points[facet.vertex_indices[i + 1]]
        for j in range(dimension):
          data_i[j] = pi[j] - p1[j]
      b = facet.normal
      b[-1] = mp.one
      A[dimension - 1, dimension - 1] = mp.one

      # Solve A x = b
      self.overwriting_solve_linear_system_of_equations(A, b)
      abs_sum = sum(abs(bi) for bi in b)
      b = [bi / abs_sum for bi in b]

      facet.offset = self.scalar_product(facet.normal, self.points[facet.vertex_indices[0]])
      self.hyper_planes += 1

      # Orient normal inwards
      if self.is_facet_visible_from_point(facet, origin):
        facet.normal = [-n for n in facet.normal]
        facet.offset = -facet.offset

  def overwriting_solve_linear_system_of_equations(self, A, b):
    n = A.cols
    assert(A.rows == n)
    assert(len(b) == n)

    # Outer product LU with partial pivoting
    # See Algorithm 3.4.1 in Golub and Van Loan - Matrix Computations, 4th Edition
    for k in range(n):
      # Determine mu with k <= mu < n so abs( A( mu, k ) ) = max( A( k:n-1, k ) )
      mu = k
      max_value = abs(A[mu, k])
      for i in range(k + 1, n):
        value = abs(A[i, k])
        if value > max_value:
          max_value = value
          mu = i

      if max_value == mp.zero:
        raise ValueError( "Singular matrix 1" )

      if k != mu:
        A.swap_row(k, mu)

      # Here, it is utilized that L is not needed
      # (if L is needed, first divide A[ i ][ k ] by A[ k ][ k ], then subtract A[ i ][ k ] * A[ k ][ j ] from A[ i ][ j ])
      inv_diag = mp.one / A[k, k]
      # size_t numElements = min( n - k - 1, size_t( 15 ) )
      data_k = A.data()[k]
      for i in range(k + 1, n):
        data_i = A.data()[i]
        factor = data_i[k] * inv_diag
        for j in range(k + 1, n):
          data_i[j] -= factor * data_k[j]

    # LU factorization completed
    # No need to solve Ly = Pb, because b = [0,...,0,1]^T, so y == Pb

    # Solve Ux = y by row-oriented back substitution
    # See Algorithm 3.1.2 in Golub and Van Loan
    for i in range(n - 1, -1, -1):
      data_i = A.data()[i]
      s = sum(data_i[j] * b[j] for j in range(i + 1, n))

      if data_i[i] != mp.zero:
        b[i] = (b[i] - s) / A[i, i]
      else:
        # Matrix is singular
        if b[i] == s:
          # U(i,i) * x(i) == 0.0 and U(i,i) == 0.0 => x(i) == 0.0 is a solution
          b[i] = mp.zero
        else:
          # U(i,i) * x(i) != 0.0 but U(i,i) == 0.0 => no solution
          raise ValueError( "Singular matrix 2" )
    # b now contains the solution x to Ax = b

  def initialize_outside_set(self, facets):
    check_point = [True] * len(self.points)
    for facet in facets:
      for i in facet.vertex_indices:
        check_point[i] = False

    for pi, point in enumerate(self.points):
      farthest_facet = None
      if check_point[pi]:
        max_distance = mp.zero
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
    self.distance_tests += 1
    return sum(starmap(mul, zip(a, b)))

  def is_facet_visible_from_point(self, facet, point):
    # Returns true if the point is contained in the open negative halfspace of the facet
    return self.scalar_product(facet.normal, point) < facet.offset

  def point_distance(self, p1, p2):
    return mp.sqrt(sum((x - y) ** 2 for x, y in zip(p1, p2)))

  def distance(self, facet, point):
    return facet.offset - self.scalar_product(facet.normal, point)

  def throw_if_not_convex_polytope(self, facets):
    if any(self.is_facet_visible_from_point(facet, neighbor.center) for facet in facets for neighbor in facet.neighbors):
      raise TypeError("Not a convex polytope")
