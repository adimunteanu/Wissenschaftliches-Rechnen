import numpy as np


####################################################################################################
# Exercise 1: Function Roots

def find_root_bisection(f: object, lival: np.floating, rival: np.floating, ival_size: np.floating = -1.0, n_iters_max: int = 256) -> np.floating:
    """
    Find a root of function f(x) in (lival, rival) with bisection method.

    Arguments:
    f: function object (assumed to be continuous), returns function value if called as f(x)
    lival: initial left boundary of interval containing root
    rival: initial right boundary of interval containing root
    ival_size: minimal size of interval / convergence criterion (optional)
    n_iters_max: maximum number of iterations (optional)

    Return:
    root: approximate root of the function
    """

    assert (n_iters_max > 0)
    assert (rival > lival)

    # set meaningful minimal interval size if not given as parameter, e.g. 10 * eps
    if ival_size == -1:
        np.finfo(np.float64).eps * 10

    # intialize iteration
    fl = f(lival)
    fr = f(rival)

    # make sure the given interval contains a root
    assert (not ((fl > 0.0 and fr > 0.0) or (fl < 0.0 and fr < 0.0)))

    n_iterations = 0
    # loop until final interval is found, stop if max iterations are reached
    while (np.abs(lival - rival) > ival_size) and (n_iterations < 256):
        x = (lival + rival) / 2
        func_val_x = f(x)

        if func_val_x < 0:
            if fl < 0:
                lival = x
            else:
                rival = x
        else:
            if fl < 0:
                rival = x
            else:
                lival = x

        fl = f(lival)
        n_iterations += 1

    # calculate final approximation to root
    root = np.float64(lival)

    return root


def find_root_newton(f: object, df: object, start: np.inexact, n_iters_max: int = 256) -> (np.inexact, int):
    """
    Find a root of f(x)/f(z) starting from start using Newton's method.

    Arguments:
    f: function object (assumed to be continuous), returns function value if called as f(x)
    df: derivative of function f, also callable
    start: start position, can be either float (for real valued functions) or complex (for complex valued functions)
    n_iters_max: maximum number of iterations (optional)

    Return:
    root: approximate root, should have the same format as the start value start
    n_iterations: number of iterations
    """

    assert(n_iters_max > 0)

    # Initialize root with start value
    root = start

    # chose meaningful convergence criterion eps, e.g 10 * eps
    convergence_criterium = np.finfo(np.float64).eps * 10

    # Initialize iteration
    fc = f(root)
    dfc = df(root)
    n_iterations = 0

    # loop until convergence criterion eps is met
    while abs(f(root)) > convergence_criterium:
        # return root and n_iters_max+1 if abs(derivative) is below f_eps or abs(root) is above 1e5 (to avoid divergence)
        if abs(dfc) < convergence_criterium or abs(root) > 1e5:
            return root, n_iters_max + 1
        # update root value and function/dfunction values
        root = root - fc / dfc
        print("function value before")
        print(fc)
        fc = f(root)
        print("function value after")
        print(fc)
        dfc = df(root)
        n_iterations += 1
        # avoid infinite loops and return (root, n_iters_max+1)
        if n_iterations > n_iters_max:
            return root, n_iters_max + 1

    return root, n_iterations

####################################################################################################
# Exercise 2: Newton Fractal


def generate_newton_fractal(f: object, df: object, roots: np.ndarray, sampling: np.ndarray, n_iters_max: int=20) -> np.ndarray:
    """
    Generates a Newton fractal for a given function and sampling data.

    Arguments:
    f: function (handle)
    df: derivative of function (handle)
    roots: array of the roots of the function f
    sampling: sampling of complex plane as 2d array
    n_iters_max: maxium number of iterations the newton method can calculate to find a root

    Return:
    result: 3d array that contains for each sample in sampling the index of the associated root and the number of iterations performed to reach it.
    """

    result = np.zeros((sampling.shape[0], sampling.shape[1], 2), dtype=int)

    # iterate over sampling grid
    for i in range(sampling.shape[0]):
        for j in range(sampling.shape[1]):
            # run Newton iteration to find a root and the iterations for the sample (in maximum n_iters_max iterations)
            found_root, n_iters = find_root_newton(f, df, sampling[i, j], n_iters_max)
            # determine the index of the closest root from the roots array. The functions np.argmin and np.tile could be helpful.
            roots_copy = np.tile(roots, 1)
            roots_copy = abs(roots_copy - found_root)
            index = np.argmin(roots_copy)

            # write the index and the number of needed iterations to the result
            result[i, j] = np.array([index, n_iters])

    return result


####################################################################################################
# Exercise 3: Minimal Surfaces

def surface_area(v: np.ndarray, f: np.ndarray) -> float:
    """
    Calculate the area of the given surface represented as triangles in f.

    Arguments:
    v: vertices of the triangles
    f: vertex indices of all triangles. f[i] gives 3 vertex indices for the three corners of the triangle i

    Return:
    area: the total surface area
    """

    # initialize area
    area = 0.0
    
    # iterate over all triangles and sum up their area
    for i in range(f.shape[0]):
        x = v[f[i][0]]
        y = v[f[i][1]]
        z = v[f[i][2]]
        xy = np.abs(np.linalg.norm(x - y))
        yz = np.abs(np.linalg.norm(y - z))
        zx = np.abs(np.linalg.norm(z - x))
        p = (xy + yz + zx) / 2
        area += (p * (p - xy) * (p - yz) * (p - zx)) ** 0.5
    
    return area


def surface_area_gradient(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Calculate the area gradient of the given surface represented as triangles in f.

    Arguments:
    v: vertices of the triangles
    f: vertex indices of all triangles. f[i] gives 3 vertex indices for the three corners of the triangle i

    Return:
    gradient: the surface area gradient of all vertices in v
    """

    # intialize the gradient
    gradient = np.zeros(v.shape)
    
    # iterate over all triangles and sum up the vertices gradients
    for i in range(f.shape[0]):
        x0 = v[f[i, 0]]
        x1 = v[f[i, 1]]
        x2 = v[f[i, 2]]

        x1x2 = x1 - x2
        x0x2 = x0 - x2
        n0 = np.cross(x1x2, x0x2) / np.linalg.norm(np.cross(x1x2, x0x2))
        gradient[f[i, 0]] -= np.cross(n0, x1x2)

        x2x0 = x2 - x0
        x1x0 = x1 - x0
        n1 = np.cross(x2x0, x1x0) / np.linalg.norm(np.cross(x2x0, x1x0))
        gradient[f[i, 1]] -= np.cross(n1, x2x0)

        x0x1 = x0 - x1
        x2x1 = x2 - x1
        n2 = np.cross(x0x1, x2x1) / np.linalg.norm(np.cross(x0x1, x2x1))
        gradient[f[i, 2]] -= np.cross(n2, x0x1)
    
    return gradient


def gradient_descent_step(v: np.ndarray, f: np.ndarray, c: np.ndarray, epsilon: float=1e-6) -> (bool, float, np.ndarray, np.ndarray):
    """
    Calculate the minimal area surface for the given triangles in v/f and boundary representation in c.

    Arguments:
    v: vertices of the triangles
    f: vertex indices of all triangles. f[i] gives 3 vertex indices for the three corners of the triangle i
    c: list of vertex indices which are fixed and can't be moved
    epsilon: difference tolerance between old area and new area

    Return:
    converged: flag that determines whether the function converged
    area: new surface area after the gradient descent step
    updated_v: vertices with changed positions
    gradient: calculated gradient
    """


    # calculate gradient and area before changing the surface
    gradient = surface_area_gradient(v, f)
    gradient_copy = np.tile(gradient, 1)
    area = surface_area(v, f)

    # TODO: calculate indices of vertices whose position can be changed
    for i in range(c.shape[0]):
        gradient[c[i]] = 0

    # TODO: find suitable step size so that area can be decreased, don't change v yet
    step = 1
    new_area_vertices = v + step * gradient

    while area < surface_area(new_area_vertices, f):
        step = step * 0.5
        new_area_vertices = v + step * gradient

    # TODO: now update vertex positions in v
    v = v + step * gradient
    # TODO: Check if new area differs only epsilon from old area
    new_area = surface_area(v, f)
    if abs(area - new_area) < epsilon:
        return True, new_area, v, gradient_copy
    # Return (True, area, v, gradient) to show that we converged and otherwise (False, area, v, gradient)

    return False, new_area, v, gradient_copy


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
