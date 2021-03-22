import matplotlib.pyplot as plt
import numpy as np


# Function taken from the tutorial_4_part3_4_hermite_splines Jupyter Notebook
def plot_spline(points_array, interpolations):
    plt.scatter(points[0], points[1], color="black")
    # Plot piecewise base polynomials
    for i in range(len(points[0]) - 1):
        # Plot local base polynomial
        p = interpolations[i]
        px = np.linspace(points_array[0][i], points_array[0][i + 1], 10000 // len(points[0]))
        py = p(px)
        plt.plot(px, py, '-')

    plt.show()


# Cleaner version of my homework implementation
def natural_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Interpolates the given function using a spline with natural boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)
    n = x.size

    V = np.zeros((4 * n - 4, 4 * n - 4))
    f = np.zeros(4 * n - 4)

    # define helper functions
    def power(k, j):
        return x[k] ** j

    def first_derivative(k, j):
        return j * x[k + 1] ** (j - 1) if j > 0 else 0

    def second_derivative(k, j):
        return (j - 1) * j * x[k + 1] ** (j - 2) if j > 1 else 0

    # Construct the system of linear equations
    for i in range(n - 1):
        V[4 * i, (i * 4):((i + 1) * 4)], f[4 * i] = np.array([power(i, j) for j in range(4)]), y[i]
        V[4 * i + 1, (i * 4):((i + 1) * 4)], f[4 * i] = np.array([power(i + 1, j) for j in range(4)]), y[i + 1]
        f[4 * i] = y[i]
        f[4 * i + 1] = y[i + 1]
        f[4 * i + 2] = 0
        f[4 * i + 3] = 0
        if i < n - 2:
            V[4 * i + 2, (i * 4):((i + 1) * 4)] = np.array([first_derivative(i, j) for j in range(4)])
            V[4 * i + 2, ((i + 1) * 4):((i + 2) * 4)] = - V[4 * i + 2, (i * 4):((i + 1) * 4)]
            V[4 * i + 3, (i * 4):((i + 1) * 4)] = np.array([second_derivative(i, j) for j in range(4)])
            V[4 * i + 3, ((i + 1) * 4):((i + 2) * 4)] = - V[4 * i + 3, (i * 4):((i + 1) * 4)]

    # Edge natural conditions
    for j in range(4):
        V[4 * (n - 2) + 2, j] = (j - 1) * j * x[0] ** (j - 2) if j > 1 else 0
        V[4 * (n - 2) + 3, j + (n - 2) * 4] = (j - 1) * j * x[n - 1] ** (j - 2) if j > 1 else 0

    # Last 2 constant vector values
    f[4 * n - 5] = y[n - 1]
    f[4 * n - 6] = y[n - 2]

    # solve linear system for the coefficients of the spline
    c = np.linalg.solve(V, f)

    splines = []
    # extract local interpolation coefficients from solution
    for i in range(n - 1):
        splines.append(np.poly1d(np.flipud(c[4 * i: 4 * i + 4])))

    return splines


# Data points taken from the tutorial_4_part3_4_hermite_splines Jupyter Notebook
x_data = np.array([0, 0.3, 0.7, 1, 1.4, 1.8, 2, 2.2, 2.7, 3, 3.4, 3.5, 4])
y_data = np.array([0, 0.5, -.8, 0.9, -0.6, 0.3, 0.1, 0, -0.2, 0.75, -0.4, 0.4, 0])

points = [x_data, y_data]
spline = natural_cubic_interpolation(x_data, y_data)
plot_spline(points, spline)

# Simulate cubic spline interpolation on sin data points
x_data = np.arange(10)
y_data = np.sin(x_data)

points = [x_data, y_data]
spline = natural_cubic_interpolation(x_data, y_data)
plot_spline(points, spline)

# Simulate cubic spline interpolation on random data points indefinitely
while True:
    x_data = np.sort(np.random.rand(10) % 10)
    y_data = np.random.rand(10) % 5

    points = [x_data, y_data]
    spline = natural_cubic_interpolation(x_data, y_data)
    plot_spline(points, spline)

# script is run by typing `python3 beispiele.py` in the terminal
