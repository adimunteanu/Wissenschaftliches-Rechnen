import numpy as np


####################################################################################################
# Exercise 1: Interpolation

def lagrange_interpolation(x: np.ndarray, y: np.ndarray) -> (np.poly1d, list):
    """
    Generate Lagrange interpolation polynomial.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    polynomial: polynomial as np.poly1d object
    base_functions: list of base polynomials
    """

    assert (x.size == y.size)

    polynomial = np.poly1d(0)
    base_functions = []

    n = x.size

    for i in range(n):
        l_i = np.poly1d(1)
        for j in range(n):
            if j != i:
                l_i *= np.poly1d([1, - x[j]])
                l_i /= (x[i] - x[j])
        polynomial += (y[i] * l_i)
        base_functions.append(l_i)

    return polynomial, base_functions




def hermite_cubic_interpolation(x: np.ndarray, y: np.ndarray, yp: np.ndarray) -> list:
    """
    Compute hermite cubic interpolation spline

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points
    yp: derivative values of interpolation points

    Returns:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size == yp.size)

    spline = []
    n = x.size

    for i in range(n - 1):
        L = np.zeros((4, 4))
        L[0] = [1, x[i], x[i]**2, x[i]**3]
        L[1] = [1, x[i+1], x[i+1]**2, x[i+1]**3]
        L[2] = [0, 1, 2 * x[i], 3 * x[i]**2]
        L[3] = [0, 1, 2 * x[i+1], 3 * x[i+1]**2]

        f = np.array([y[i], y[i+1], yp[i], yp[i+1]])

        c = np.linalg.solve(L, f)
        c = np.flipud(c)
        spline.append(np.poly1d(c))

    return spline



####################################################################################################
# Exercise 2: Animation

def natural_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Intepolate the given function using a spline with natural boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)
    # TODO construct linear system with natural boundary conditions

    # TODO solve linear system for the coefficients of the spline

    spline = []
    # TODO extract local interpolation coefficients from solution


    return spline


def periodic_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Interpolate the given function with a cubic spline and periodic boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)
    # TODO: construct linear system with periodic boundary conditions

    # TODO solve linear system for the coefficients of the spline

    spline = []
    # TODO extract local interpolation coefficients from solution


    return spline


if __name__ == '__main__':

    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
