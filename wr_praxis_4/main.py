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
    n = x.size

    V = np.zeros((4 * n - 4, 4 * n - 4))
    f = np.zeros(4 * n - 4)

    for i in range(n):
        if i == 0:
            # first row
            V[0, 0] = 1
            V[0, 1] = x[i]
            V[0, 2] = x[i]**2
            V[0, 3] = x[i]**3
            # second last row
            V[4 * n - 6, 2] = 2
            V[4 * n - 6, 3] = 6 * x[i]

            f[0] = y[0]
        elif i == n - 1:
            # third last row
            V[4 * n - 7, 4 * n - 8] = 1
            V[4 * n - 7, 4 * n - 7] = x[i]
            V[4 * n - 7, 4 * n - 6] = x[i]**2
            V[4 * n - 7, 4 * n - 5] = x[i]**3

            # last row
            V[4 * n - 5, 4 * n - 6] = 2
            V[4 * n - 5, 4 * n - 5] = 6 * x[i]

            f[4 * n - 7] = y[n - 1]
        else:
            V[4 * (i - 1) + 1, 4 * (i - 1)] = 1
            V[4 * (i - 1) + 1, 4 * (i - 1) + 1] = x[i]
            V[4 * (i - 1) + 1, 4 * (i - 1) + 2] = x[i]**2
            V[4 * (i - 1) + 1, 4 * (i - 1) + 3] = x[i]**3

            V[4 * (i - 1) + 2, 4 * (i - 1)] = 0
            V[4 * (i - 1) + 2, 4 * (i - 1) + 1] = 1
            V[4 * (i - 1) + 2, 4 * (i - 1) + 2] = 2 * x[i]
            V[4 * (i - 1) + 2, 4 * (i - 1) + 3] = 3 * x[i]**2
            V[4 * (i - 1) + 2, 4 * (i - 1) + 4] = 0
            V[4 * (i - 1) + 2, 4 * (i - 1) + 5] = -1
            V[4 * (i - 1) + 2, 4 * (i - 1) + 6] = -2 * x[i]
            V[4 * (i - 1) + 2, 4 * (i - 1) + 7] = -3 * x[i]**2

            V[4 * (i - 1) + 3, 4 * (i - 1)] = 0
            V[4 * (i - 1) + 3, 4 * (i - 1) + 1] = 0
            V[4 * (i - 1) + 3, 4 * (i - 1) + 2] = 2
            V[4 * (i - 1) + 3, 4 * (i - 1) + 3] = 6 * x[i]
            V[4 * (i - 1) + 3, 4 * (i - 1) + 4] = 0
            V[4 * (i - 1) + 3, 4 * (i - 1) + 5] = 0
            V[4 * (i - 1) + 3, 4 * (i - 1) + 6] = -2
            V[4 * (i - 1) + 3, 4 * (i - 1) + 7] = -6 * x[i]

            V[4 * (i - 1) + 4, 4 * i] = 1
            V[4 * (i - 1) + 4, 4 * i + 1] = x[i]
            V[4 * (i - 1) + 4, 4 * i + 2] = x[i]**2
            V[4 * (i - 1) + 4, 4 * i + 3] = x[i]**3

            f[4 * (i - 1) + 1] = y[i]
            f[4 * (i - 1) + 4] = y[i]


    # TODO solve linear system for the coefficients of the spline
    c = np.linalg.solve(V, f)

    spline = []
    # TODO extract local interpolation coefficients from solution
    for i in range(n - 1):
        spline.append(np.poly1d(np.flipud(c[4 * i: 4 * i + 4])))

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
    n = x.size

    V = np.zeros((4 * n - 4, 4 * n - 4))
    f = np.zeros(4 * n - 4)

    for i in range(n):
        if i == 0:
            # first row
            V[0, 0] = 1
            V[0, 1] = x[i]
            V[0, 2] = x[i] ** 2
            V[0, 3] = x[i] ** 3
            # second last row
            V[4 * n - 6, 0] = 0
            V[4 * n - 6, 1] = 1
            V[4 * n - 6, 2] = 2 * x[i]
            V[4 * n - 6, 3] = 3 * x[i]**2

            # last row
            V[4 * n - 5, 0] = 0
            V[4 * n - 5, 1] = 0
            V[4 * n - 5, 2] = 2
            V[4 * n - 5, 3] = 6 * x[i]


            f[0] = y[0]
        elif i == n - 1:
            # third last row
            V[4 * n - 7, 4 * n - 8] = 1
            V[4 * n - 7, 4 * n - 7] = x[i]
            V[4 * n - 7, 4 * n - 6] = x[i]**2
            V[4 * n - 7, 4 * n - 5] = x[i]**3

            # second last row
            V[4 * n - 6, 4 * n - 8] = 0
            V[4 * n - 6, 4 * n - 7] = -1
            V[4 * n - 6, 4 * n - 6] = -2 * x[i]
            V[4 * n - 6, 4 * n - 5] = -3 * x[i]**2

            # last row
            V[4 * n - 5, 4 * n - 8] = 0
            V[4 * n - 5, 4 * n - 7] = 0
            V[4 * n - 5, 4 * n - 6] = -2
            V[4 * n - 5, 4 * n - 5] = -6 * x[i]

            f[4 * n - 7] = y[n - 1]
        else:
            V[4 * (i - 1) + 1, 4 * (i - 1)] = 1
            V[4 * (i - 1) + 1, 4 * (i - 1) + 1] = x[i]
            V[4 * (i - 1) + 1, 4 * (i - 1) + 2] = x[i] ** 2
            V[4 * (i - 1) + 1, 4 * (i - 1) + 3] = x[i] ** 3

            V[4 * (i - 1) + 2, 4 * (i - 1)] = 0
            V[4 * (i - 1) + 2, 4 * (i - 1) + 1] = 1
            V[4 * (i - 1) + 2, 4 * (i - 1) + 2] = 2 * x[i]
            V[4 * (i - 1) + 2, 4 * (i - 1) + 3] = 3 * x[i] ** 2
            V[4 * (i - 1) + 2, 4 * (i - 1) + 4] = 0
            V[4 * (i - 1) + 2, 4 * (i - 1) + 5] = -1
            V[4 * (i - 1) + 2, 4 * (i - 1) + 6] = -2 * x[i]
            V[4 * (i - 1) + 2, 4 * (i - 1) + 7] = -3 * x[i] ** 2

            V[4 * (i - 1) + 3, 4 * (i - 1)] = 0
            V[4 * (i - 1) + 3, 4 * (i - 1) + 1] = 0
            V[4 * (i - 1) + 3, 4 * (i - 1) + 2] = 2
            V[4 * (i - 1) + 3, 4 * (i - 1) + 3] = 6 * x[i]
            V[4 * (i - 1) + 3, 4 * (i - 1) + 4] = 0
            V[4 * (i - 1) + 3, 4 * (i - 1) + 5] = 0
            V[4 * (i - 1) + 3, 4 * (i - 1) + 6] = -2
            V[4 * (i - 1) + 3, 4 * (i - 1) + 7] = -6 * x[i]

            V[4 * (i - 1) + 4, 4 * i] = 1
            V[4 * (i - 1) + 4, 4 * i + 1] = x[i]
            V[4 * (i - 1) + 4, 4 * i + 2] = x[i] ** 2
            V[4 * (i - 1) + 4, 4 * i + 3] = x[i] ** 3

            f[4 * (i - 1) + 1] = y[i]
            f[4 * (i - 1) + 4] = y[i]

    # TODO solve linear system for the coefficients of the spline
    c = np.linalg.solve(V, f)

    spline = []
    # TODO extract local interpolation coefficients from solution
    for i in range(n - 1):
        spline.append(np.poly1d(np.flipud(c[4 * i: 4 * i + 4])))

    return spline


if __name__ == '__main__':

    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
