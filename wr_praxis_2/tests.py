
import numpy as np
import matplotlib.pyplot as plt
import datetime

import unittest
import tomograph
from main import compute_tomograph, gaussian_elimination, back_substitution, compute_cholesky, solve_cholesky


class Tests(unittest.TestCase):
    def test_gaussian_elimination(self):
        A = np.random.randn(4, 4)
        x = np.random.rand(4)
        b = np.dot(A, x)
        A_elim, b_elim = gaussian_elimination(A, b)
        self.assertTrue(np.allclose(np.linalg.solve(A_elim, b_elim), x))  # Check if system is still solvable
        self.assertTrue(np.allclose(A_elim, np.triu(A_elim)))  # Check if matrix is upper triangular

        # Pivot required
        A = np.array([[0, 1], [1, 0]])
        x = np.array([0, 1])
        b = np.dot(A, x)
        A_elim, b_elim = gaussian_elimination(A, b)
        print(A_elim)
        print(b_elim)
        self.assertTrue(np.allclose(np.linalg.solve(A_elim, b_elim), x))  # Check if system is still solvable

        A = np.array([[0, 2, 2], [1, 3, 2], [3, 3, 3]])
        b = np.array([3, 2, 3])
        x = np.linalg.solve(A, b);
        A_elim, b_elim = gaussian_elimination(A, b)
        print(A_elim)
        print(b_elim)
        self.assertTrue(np.allclose(np.linalg.solve(A_elim, b_elim), x))  # Check if system is still solvable


    def test_back_substitution(self):
        A = np.random.randn(4, 4)
        x = np.random.rand(4)
        b = np.dot(A, x)
        A_elim, b_elim = gaussian_elimination(A, b)
        self.assertTrue(np.allclose(np.linalg.solve(A, b), back_substitution(A_elim, b_elim)))

    def test_cholesky_decomposition(self):
        # A = np.random.randn(4, 4)
        #
        # M = np.dot(A, A.transpose())
        M = np.array([[5, 5, 5], [5, 5, 5], [5, 5, 5]])
        self.assertTrue(np.allclose(np.linalg.cholesky(M), compute_cholesky(M)))

    def test_solve_cholesky(self):
        A = np.random.randn(4, 4)
        M = np.dot(A, A.transpose())
        x = np.random.randn(4)
        b = np.dot(M, x)
        L = np.linalg.cholesky(M)
        self.assertTrue(np.allclose(np.linalg.solve(M, b), solve_cholesky(L, b)))

    def test_compute_tomograph(self):
        t = datetime.datetime.now()
        print("Start time: " + str(t.hour) + ":" + str(t.minute) + ":" + str(t.second))

        # Compute tomographic image
        n_shots = 64  # 128
        n_rays = 64  # 128
        n_grid = 32  # 64
        tim = compute_tomograph(n_shots, n_rays, n_grid)

        t = datetime.datetime.now()
        print("End time: " + str(t.hour) + ":" + str(t.minute) + ":" + str(t.second))

        # Visualize image
        plt.imshow(tim, cmap='gist_yarg', extent=[-1.0, 1.0, -1.0, 1.0],
                   origin='lower', interpolation='nearest')
        plt.gca().set_xticks([-1, 0, 1])
        plt.gca().set_yticks([-1, 0, 1])
        plt.gca().set_title('%dx%d' % (n_grid, n_grid))

        plt.show()


if __name__ == '__main__':
    unittest.main()

