import unittest
from src.basic_functions import gamma, v_given_gamma, multi_root


class TestBasicFuncs(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_gamma_to_v_and_back(self):
        for v in [1000, 2000, 3000]:
            self.assertAlmostEqual(v, v_given_gamma(gamma(v)), delta=10e-2)

    def test_multiple_roots(self):
        f = lambda x: x * (x - 1) * (x + 1)
        roots = multi_root(f, [-2, 2])
        self.assertListEqual(list(roots), [-1, 0, 1])

    def test_mult_roots_different(self):
        f = lambda x: x * (x - 1) * (x - 2) * (x - 3)
        roots = multi_root(f, [-1, 4])
        for my_root, test_root in zip(roots, [0, 1, 2, 3]):
            self.assertAlmostEqual(my_root, test_root, delta=10e-8)


