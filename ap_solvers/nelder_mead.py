import copy
from mpmath import mp
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
from collections import OrderedDict


@dataclass
class NelderMeadState:
    """State of the Nelder-Mead optimization algorithm."""
    simplex: List[List]  # List of [point, value] pairs
    iteration: int
    no_improvement_count: int
    previous_best: mp.mpf
    dim: int


class LRUCache:
    """Simple LRU cache for function evaluations."""
    def __init__(self, maxsize=1000):
        self.cache = OrderedDict()
        self.maxsize = maxsize
    
    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def set(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)


def _initialize_simplex(f_cached: Callable, x_start: List, step: float, dim: int) -> List[List]:
    """Initialize the simplex with n+1 vertices."""
    simplex = [[x_start, f_cached(x_start)]]
    
    for i in range(dim):
        x = copy.copy(x_start)
        x[i] = x[i] + step
        score = f_cached(x)
        simplex.append([x, score])
    
    return simplex


def _compute_centroid(simplex: List[List], dim: int) -> List:
    """Compute centroid of all vertices except the worst."""
    centroid = [mp.zero] * dim
    n_vertices = len(simplex) - 1
    
    for vertex in simplex[:-1]:
        for i, coord in enumerate(vertex[0]):
            centroid[i] += coord / mp.mpf(n_vertices)
    
    return centroid


def _replace_worst(simplex: List[List], new_point: List, new_value) -> List[List]:
    """Replace the worst vertex in the simplex."""
    return simplex[:-1] + [[new_point, new_value]]


def _check_convergence(state: NelderMeadState, no_improve_thr, no_improv_break: int, 
                       max_iter: int) -> Tuple[bool, Optional[str]]:
    """
    Check if the algorithm has converged.
    
    Returns:
        tuple: (converged, status) where converged is bool and status is termination reason
    """
    # Check maximum iterations
    if max_iter and state.iteration >= max_iter:
        return True, "Maximum number of iterations"
    
    # Check for no improvement
    if state.no_improvement_count >= no_improv_break:
        return True, "Optimal solution found"
    
    return False, None


def _print_iteration_info(iteration: int, best_value, best_point: List, 
                         no_improvement_count: int):
    """Print information about current iteration."""
    point_str = "[" + ", ".join(f"{float(x):.6f}" for x in best_point[:3])
    if len(best_point) > 3:
        point_str += ", ..."
    point_str += "]"
    
    print(f"Iteration {iteration:4d} | Best value: {float(best_value):12.6e} | "
          f"Best point: {point_str} | No improve: {no_improvement_count}")


def nelder_mead(f, x_start,
                step=0.05, no_improve_thr=mp.mpf('1e-50'),
                no_improv_break=15, max_iter=0,
                alpha=1., gamma=2., rho=-0.5, sigma=0.5,
                verbose=False, max_cache_size=1000):
    '''
    Minimize a scalar function using the Nelder-Mead simplex algorithm.
    
    The Nelder-Mead method is a derivative-free optimization algorithm that maintains
    a simplex of n+1 points in n-dimensional space. At each iteration, it reflects,
    expands, contracts, or shrinks the simplex to find better points.
    
    Parameters:
        f (callable): Objective function to minimize. Must accept a list/array of length n
                     and return a scalar value.
        x_start (list): Initial point (list of mpf values) of length n.
        step (float): Initial simplex step size for each dimension. Default 0.05.
        no_improve_thr (mpf): Threshold for detecting improvement. If the best value 
                             improves by less than this for no_improv_break iterations,
                             the algorithm terminates. Default 1e-50.
        no_improv_break (int): Number of iterations without significant improvement before
                              terminating. Default 15.
        max_iter (int): Maximum number of iterations. Set to 0 for no limit. Default 0.
        alpha (float): Reflection coefficient. Default 1.0.
        gamma (float): Expansion coefficient. Default 2.0.
        rho (float): Contraction coefficient. Default -0.5.
        sigma (float): Shrinkage coefficient. Default 0.5.
        verbose (bool): If True, print iteration information. Default False.
        max_cache_size (int): Maximum size of function evaluation cache. Default 1000.
    
    Returns:
        tuple: (x_best, f_best, status) where:
            - x_best (list): Best point found
            - f_best (mpf): Function value at best point
            - status (str): Termination reason ("Optimal solution found" or 
                          "Maximum number of iterations")
    
    Reference:
        https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
    '''

    # Function evaluation cache with LRU eviction
    cache = LRUCache(maxsize=max_cache_size)
    
    def cached_f(x):
        x_tuple = tuple(x)
        cached_value = cache.get(x_tuple)
        if cached_value is not None:
            return cached_value
        value = f(x)
        cache.set(x_tuple, value)
        return value
    
    # Initialize state
    dim = len(x_start)
    initial_simplex = _initialize_simplex(cached_f, x_start, step, dim)
    
    state = NelderMeadState(
        simplex=initial_simplex,
        iteration=0,
        no_improvement_count=0,
        previous_best=initial_simplex[0][1],  # Will be updated after first sort
        dim=dim
    )
    
    if verbose:
        print("=" * 80)
        print("Starting Nelder-Mead Optimization")
        print(f"Dimension: {dim}, Max iterations: {max_iter if max_iter else 'unlimited'}")
        print("=" * 80)

    if verbose:
        print("=" * 80)
        print("Starting Nelder-Mead Optimization")
        print(f"Dimension: {dim}, Max iterations: {max_iter if max_iter else 'unlimited'}")
        print("=" * 80)

    # Main optimization loop
    while True:
        # Sort vertices by function value (best to worst)
        state.simplex.sort(key=lambda vertex: vertex[1])
        best_value = state.simplex[0][1]
        
        # Check for convergence
        converged, status = _check_convergence(state, no_improve_thr, no_improv_break, max_iter)
        if converged:
            if verbose:
                print("=" * 80)
                print(f"Optimization terminated: {status}")
                print(f"Total iterations: {state.iteration}")
                print(f"Final best value: {float(best_value):.12e}")
                print("=" * 80)
            return state.simplex[0][0], state.simplex[0][1], status
        
        # Update iteration counter
        state.iteration += 1

        # Update improvement tracking
        if best_value < state.previous_best - no_improve_thr:
            state.no_improvement_count = 0
            state.previous_best = best_value
        else:
            state.no_improvement_count += 1
        
        # Print iteration info
        if verbose:
            _print_iteration_info(state.iteration, best_value, state.simplex[0][0], 
                                state.no_improvement_count)

        # Compute centroid of all vertices except the worst
        centroid = _compute_centroid(state.simplex, state.dim)

        # Reflection: try reflecting worst point through centroid
        reflected = [c + alpha * (c - w) for c, w in zip(centroid, state.simplex[-1][0])]
        reflected_score = cached_f(reflected)
        
        if state.simplex[0][1] <= reflected_score < state.simplex[-2][1]:
            state.simplex = _replace_worst(state.simplex, reflected, reflected_score)
            continue

        # Expansion: if reflected point is best, try expanding further
        if reflected_score < state.simplex[0][1]:
            expanded = [c + gamma * (c - w) for c, w in zip(centroid, state.simplex[-1][0])]
            expanded_score = cached_f(expanded)
            
            if expanded_score < reflected_score:
                state.simplex = _replace_worst(state.simplex, expanded, expanded_score)
            else:
                state.simplex = _replace_worst(state.simplex, reflected, reflected_score)
            continue

        # Contraction: if reflection didn't help, contract toward centroid
        contracted = [c + rho * (c - w) for c, w in zip(centroid, state.simplex[-1][0])]
        contracted_score = cached_f(contracted)
        
        if contracted_score < state.simplex[-1][1]:
            state.simplex = _replace_worst(state.simplex, contracted, contracted_score)
            continue

        # Shrinkage: contract all vertices toward the best vertex
        best_vertex = state.simplex[0][0]
        new_simplex = []
        for vertex in state.simplex:
            shrunk = [b + sigma * (v - b) for b, v in zip(best_vertex, vertex[0])]
            score = cached_f(shrunk)
            new_simplex.append([shrunk, score])
        state.simplex = new_simplex