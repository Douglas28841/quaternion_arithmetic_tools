import math
import unittest

from quaternion_arithmetic_tools import (
    conjugate,
    hamilton_product,
    norm,
    normalize,
    rotate_vector,
)


class TestHamiltonProduct(unittest.TestCase):
    def test_identity(self):
        identity = (1.0, 0.0, 0.0, 0.0)
        q = (0.5, -0.5, 0.25, 0.75)
        self.assertEqual(hamilton_product(identity, q), q)
        self.assertEqual(hamilton_product(q, identity), q)

    def test_non_commutative(self):
        i = (0.0, 1.0, 0.0, 0.0)
        j = (0.0, 0.0, 1.0, 0.0)
        self.assertEqual(hamilton_product(i, j), (0.0, 0.0, 0.0, 1.0))
        self.assertEqual(hamilton_product(j, i), (0.0, 0.0, 0.0, -1.0))

    def test_known_values(self):
        a = (1.0, 2.0, 3.0, 4.0)
        b = (5.0, 6.0, 7.0, 8.0)
        self.assertEqual(
            hamilton_product(a, b),
            (-60.0, 12.0, 30.0, 24.0),
        )

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            hamilton_product((1.0, 2.0, 3.0), (1.0, 0.0, 0.0, 0.0))
        with self.assertRaises(TypeError):
            hamilton_product((1.0, 2.0, 3.0, "4"), (1.0, 0.0, 0.0, 0.0))


class TestConjugate(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(conjugate((1.0, 2.0, 3.0, 4.0)), (1.0, -2.0, -3.0, -4.0))

    def test_involution(self):
        q = (0.5, -1.5, 2.5, -3.5)
        self.assertEqual(conjugate(conjugate(q)), q)


class TestNorm(unittest.TestCase):
    def test_unit_quaternion(self):
        self.assertAlmostEqual(norm((1.0, 0.0, 0.0, 0.0)), 1.0)

    def test_known_value(self):
        self.assertAlmostEqual(norm((1.0, 2.0, 2.0, 4.0)), 5.0)

    def test_zero(self):
        self.assertEqual(norm((0.0, 0.0, 0.0, 0.0)), 0.0)


class TestNormalize(unittest.TestCase):
    def test_unit_result(self):
        q = (1.0, 2.0, 3.0, 4.0)
        n = normalize(q)
        self.assertAlmostEqual(norm(n), 1.0)

    def test_direction_preserved(self):
        q = (0.0, 3.0, 0.0, 4.0)
        n = normalize(q)
        self.assertAlmostEqual(n[1], 0.6)
        self.assertAlmostEqual(n[3], 0.8)

    def test_zero_raises(self):
        with self.assertRaises(ValueError):
            normalize((0.0, 0.0, 0.0, 0.0))


class TestRotateVector(unittest.TestCase):
    def test_identity_rotation(self):
        q = (1.0, 0.0, 0.0, 0.0)
        v = (1.0, 2.0, 3.0)
        self.assertEqual(rotate_vector(q, v), v)

    def test_90_degrees_about_z(self):
        # Rotation by +90 degrees about the z axis maps (1,0,0) to (0,1,0).
        s = math.sqrt(0.5)
        q = (s, 0.0, 0.0, s)
        got = rotate_vector(q, (1.0, 0.0, 0.0))
        self.assertAlmostEqual(got[0], 0.0)
        self.assertAlmostEqual(got[1], 1.0)
        self.assertAlmostEqual(got[2], 0.0)

    def test_180_degrees_about_x(self):
        q = (0.0, 1.0, 0.0, 0.0)
        got = rotate_vector(q, (0.0, 1.0, 0.0))
        self.assertAlmostEqual(got[0], 0.0)
        self.assertAlmostEqual(got[1], -1.0)
        self.assertAlmostEqual(got[2], 0.0)

    def test_non_unit_scales_by_norm_squared(self):
        # q = 2 * identity, so the result is 4 * v.
        q = (2.0, 0.0, 0.0, 0.0)
        got = rotate_vector(q, (1.0, 2.0, 3.0))
        self.assertEqual(got, (4.0, 8.0, 12.0))

    def test_rejects_bad_vector(self):
        q = (1.0, 0.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            rotate_vector(q, (1.0, 2.0))


if __name__ == "__main__":
    unittest.main()
