
import numpy as np
import tomograph


####################################################################################################
# Exercise 1: Gaussian elimination

def gaussian_elimination(A: np.ndarray, b: np.ndarray, use_pivoting: bool = True) -> (np.ndarray, np.ndarray):
    """
    Gaussian Elimination of Ax=b with or without pivoting.

    Arguments:
    A : matrix, representing left side of equation system of size: (m,m)
    b : vector, representing right hand side of size: (m, )
    use_pivoting : flag if pivoting should be used

    Return:
    A : reduced result matrix in row echelon form (type: np.ndarray, size: (m,m))
    b : result vector in row echelon form (type: np.ndarray, size: (m, ))

    Raised Exceptions:
    ValueError: if matrix and vector sizes are incompatible, matrix is not square or pivoting is disabled but necessary

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """
    # Create copies of input matrix and vector to leave them unmodified
    A = A.copy()
    b = b.copy()

    # TODO: Test if shape of matrix and vector is compatible and raise ValueError if not
    if A.shape[0] != A.shape[1]:
        raise ValueError('Matrix is not square!')
    elif A.shape[0] != b.shape[0]:
        raise ValueError('Matrix and vector size are incompatible!')

    # TODO: Perform gaussian elimination
    i = 0
    m = b.shape[0]

    while i < m:
        if use_pivoting:
            # implementation of pivoting
            column_copy = A[i:m, i:i + 1].copy()
            max_index = np.abs(column_copy).argmax()

            if column_copy.flat[max_index] == 0:
                raise ValueError("System has none or infinite solutions")

            if max_index + i != i:
                A[[i, i + max_index]] = A[[i + max_index, i]]
                b[[i, i + max_index]] = b[[i + max_index, i]]
        else:
            if A[i, i] == 0:
                raise ValueError("Pivoting is disabled but is required!")

        # implementation of elimination
        for j in range(i + 1, m):
            k = A[j, i] / A[i, i]
            A[j] = A[j] - k * A[i]
            b[j] = b[j] - k * b[i]
        i += 1

    return A, b


def back_substitution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Back substitution for the solution of a linear system in row echelon form.

    Arguments:
    A : matrix in row echelon representing linear system
    b : vector, representing right hand side

    Return:
    x : solution of the linear system

    Raised Exceptions:
    ValueError: if matrix/vector sizes are incompatible or no/infinite solutions exist

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """

    # TODO: Test if shape of matrix and vector is compatible and raise ValueError if not
    if A.shape[0] != A.shape[1]:
        raise ValueError('Matrix is not square!')
    elif A.shape[0] != b.shape[0]:
        raise ValueError('Matrix and vector size are incompatible!')

    # TODO: Initialize solution vector with proper size
    m = b.shape[0]
    x = np.zeros(m)

    # TODO: Run backsubstitution and fill solution vector, raise ValueError if no/infinite solutions exist
    i = m - 1

    while i >= 0:
        if A[i, i] == 0:
            raise ValueError("System has none or infinite solutions")
        else:
            x[i] = b[i]
            for j in range(i + 1, m):
                x[i] -= A[i, j] * x[j]
            x[i] /= A[i, i]
            i -= 1

    return x


####################################################################################################
# Exercise 2: Cholesky decomposition

def compute_cholesky(M: np.ndarray) -> np.ndarray:
    """
    Compute Cholesky decomposition of a matrix

    Arguments:
    M : matrix, symmetric and positive (semi-)definite

    Raised Exceptions:
    ValueError: L is not symmetric and psd

    Return:
    L :  Cholesky factor of M

    Forbidden:
    - numpy.linalg.*
    """

    # TODO check for symmetry and raise an exception of type ValueError
    (n, m) = M.shape
    if n != m:
        raise ValueError("Matrix is not quadratic!")

    if not np.allclose(M, M.T, rtol=1e-05, atol=1e-08):
        raise ValueError("Matrix is not symmetric!")


    # TODO build the factorization and raise a ValueError in case of a non-positive definite input matrix

    L = np.zeros((n, n))

    for i in range(0, n):
        for j in range(0, i + 1):
            if i == j: # diagonal element calculation
                quadratic_sum = 0
                for k in range(0, i):
                    quadratic_sum += L[i, k] * L[i, k]
                L[i, i] = M[i, i] - quadratic_sum

                if L[i, i] < 0:
                    raise ValueError("Matrix is not PSD")

                L[i, i] = np.sqrt(L[i, i])
            else: # other elements calculation
                if L[j, j] == 0:
                    raise ValueError("Matrix is not PSD")

                produkt_sum = 0
                for k in range(0, j):
                    produkt_sum += L[i, k] * L[j, k]

                L[i, j] = (M[i, j] - produkt_sum) / L[j, j]

    return L


def solve_cholesky(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve the system L L^T x = b where L is a lower triangular matrix

    Arguments:
    L : matrix representing the Cholesky factor
    b : right hand side of the linear system

    Raised Exceptions:
    ValueError: sizes of L, b do not match
    ValueError: L is not lower triangular matrix

    Return:
    x : solution of the linear system

    Forbidden:
    - numpy.linalg.*
    """

    # TODO Check the input for validity, raising a ValueError if this is not the case
    (n, m) = L.shape
    if n != m:
        raise ValueError("Matrix isn't quadratic!")

    p = b.shape[0]
    if n != p:
        raise ValueError("Dimensions of L and b don't match!")

    if not np.allclose(L, np.tril(L)):
        raise ValueError("L is not a lower triangular matrix!")

    # TODO Solve the system by forward- and backsubstitution
    y = np.zeros(n)

    # forward substitution
    for i in range(0, n):
        if L[i, i] == 0:
            raise ValueError("Infinite or no solutions")
        else:
            y[i] = b[i]
            for j in range(0, i):
                y[i] -= L[i, j] * y[j]
            y[i] /= L[i, i]

    # back substitution
    return back_substitution(L.T, y)


####################################################################################################
# Exercise 3: Tomography

def setup_system_tomograph(n_shots: np.int, n_rays: np.int, n_grid: np.int) -> (np.ndarray, np.ndarray):
    """
    Set up the linear system describing the tomographic reconstruction

    Arguments:
    n_shots  : number of different shot directions
    n_rays   : number of parallel rays per direction
    n_grid   : number of cells of grid in each direction, in total n_grid*n_grid cells

    Return:
    L : system matrix
    g : measured intensities

    Raised Exceptions:
    -

    Side Effects:
    -

    Forbidden:
    -
    """

    # TODO: Initialize system matrix with proper size
    L = np.zeros((n_shots * n_rays, n_grid * n_grid))
    # TODO: Initialize intensity vector
    g = np.zeros(n_shots * n_rays)

    # TODO: Iterate over equispaced angles, take measurements, and update system matrix and sinogram
    theta = 0
    theta_increment = np.pi / n_shots

    for i in range(0, n_shots):
        # Take a measurement with the tomograph from direction r_theta.
        # intensities: measured intensities for all <n_rays> rays of the measurement. intensities[n] contains the
        # intensity for the n-th ray
        # ray_indices: indices of rays that intersect a cell
        # isect_indices: indices of intersected cells
        # lengths: lengths of segments in intersected cells
        # The tuple (ray_indices[n], isect_indices[n], lengths[n]) stores which ray has intersected which cell with
        # which length. n runs from 0 to the amount of ray/cell intersections (-1) of this measurement.
        intensities, ray_indices, isect_indices, lengths = tomograph.take_measurement(n_grid, n_rays, theta)
        theta += theta_increment

        offset = i * n_rays

        for j in range(0, n_rays):
            g[j + offset] = intensities[j]

        for j in range(0, len(ray_indices)):
            L[ray_indices[j] + offset, isect_indices[j]] = lengths[j]

    return [L, g]


def compute_tomograph(n_shots: np.int, n_rays: np.int, n_grid: np.int) -> np.ndarray:
    """
    Compute tomographic image

    Arguments:
    n_shots  : number of different shot directions
    n_rays   : number of parallel rays per direction
    n_grid   : number of cells of grid in each direction, in total n_grid*n_grid cells

    Return:
    tim : tomographic image

    Raised Exceptions:
    -

    Side Effects:
    -

    Forbidden:
    """

    # Setup the system describing the image reconstruction
    [L, g] = setup_system_tomograph(n_shots, n_rays, n_grid)

    # TODO: Solve for tomographic image using your Cholesky solver
    # (alternatively use Numpy's Cholesky implementation)

    A_t_A = np.dot(L.T, L)
    A_t_b = np.dot(L.T, g)
    #c = solve_cholesky(compute_cholesky(A_t_A), A_t_b)
    c = solve_cholesky(np.linalg.cholesky(A_t_A), A_t_b)

    # TODO: Convert solution of linear system to 2D image
    tim = np.reshape(c, (n_grid, n_grid))

    return tim


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
