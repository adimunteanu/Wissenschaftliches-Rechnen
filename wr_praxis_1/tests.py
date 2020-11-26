import numpy as np
import unittest
from main import rotation_matrix, matrix_multiplication, compare_multiplication, inverse_rotation, machine_epsilon


class Tests(unittest.TestCase):

    def test_matrix_multiplication(self):
        a = np.random.randn(2, 2)
        c = np.random.randn(3, 3)
        self.assertTrue(np.allclose(np.dot(a, a), matrix_multiplication(a, a)))
        self.assertRaises(ValueError, matrix_multiplication, a, c)

    def test_compare_multiplication(self):
        r_dict = compare_multiplication(200, 40)
        for r in zip(r_dict["results_numpy"], r_dict["results_mat_mult"]):
            self.assertTrue(np.allclose(r[0], r[1]))

    def test_machine_epsilon(self):
        self.assertEqual(np.finfo(np.dtype(np.float32)).eps, machine_epsilon(np.dtype(np.float32)))
        self.assertEqual(np.finfo(np.dtype(np.float64)).eps, machine_epsilon(np.dtype(np.float64)))

    def test_rotation_matrix(self):
        R90 = np.array([[0, -1],
                        [1, 0]])
        R180 = np.array([[-1, 0],
                         [0, -1]])
        R270 = np.array([[0, 1],
                         [-1, 0]])
        self.assertTrue(np.allclose(R90, rotation_matrix(90)))
        self.assertTrue(np.allclose(R180, rotation_matrix(180)))
        self.assertTrue(np.allclose(R270, rotation_matrix(270)))
        pass

    def test_inverse_rotation(self):
        ran_theta = np.random.rand() * 360
        r = rotation_matrix(ran_theta)
        r_inv = inverse_rotation(ran_theta)
        self.assertTrue(np.allclose(r_inv, np.linalg.inv(r)))


if __name__ == '__main__':
    unittest.main()
