#!/usr/bin/env python3
"""CPU checks for the independent oracle and binary replay contract."""
import struct
import unittest
import numpy as np
from test_layer import synthetic, reference, validate_model, layer_params, dot2, compare
from test_k import params as existing_params


class ReferenceTests(unittest.TestCase):
    def test_existing_parameter_abi(self):
        e, weights, _ = synthetic(160, 8, 1)
        validate_model(e, weights)

        class Params:
            def i32x2(self, offset):
                return {0: (11, 9), 56: (4, 3)}[offset]

        expected, _ = existing_params(Params(), e)
        actual = layer_params(e, 11, 9, 4, 3)
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 448)
        self.assertEqual(struct.unpack_from('<4i', actual), (11, 9, 4, 3))

    def test_zero_network_preserves_shifted_input(self):
        e, weights, rng = synthetic(32, 2, 4)
        weights[:] = 0
        x = rng.normal(0, .5, (9, 11, 32)).astype(np.float16)
        actual = reference(x, e, weights, 4, 3)
        np.testing.assert_array_equal(actual, x)

    def test_bias_residual_order(self):
        e, weights, rng = synthetic(32, 2, 5)
        weights[:] = 0
        weights[e['bo']:e['bo']+32] = .03
        weights[e['b2']:e['b2']+32] = -.02
        x = rng.normal(0, .5, (8, 8, 32)).astype(np.float16)
        expected = (x + np.float16(.03)).astype(np.float16) + np.float16(-.02)
        np.testing.assert_array_equal(reference(x, e, weights, 0, 0), expected)

    def test_dot2_orientation_and_accumulator(self):
        a = np.array([[1, 2, 3, 4], [-1, 0, 2, 1]], np.float16)
        b = np.array([[2, 3], [4, 5], [6, 7], [8, 9]], np.float16)
        acc = np.array([[.5, .25], [1, 2]], np.float32)
        np.testing.assert_array_equal(dot2(a, b, acc), a.astype(np.float32) @ b.astype(np.float32) + acc)

    def test_reject_wrong_model_or_output(self):
        e, weights, _ = synthetic(64, 2, 3)
        e['kind'] = 'enc'
        with self.assertRaises(ValueError):
            validate_model(e, weights)
        x = np.ones((8, 8, 64), np.float16)
        with self.assertRaises(ValueError):
            compare(x, np.zeros_like(x), 60)
        with self.assertRaises(ValueError):
            compare(x, np.full_like(x, np.nan), 60)


if __name__ == '__main__':
    unittest.main()
