from unittest import TestCase

import numpy as np

from network import Network


class TestNetwork(TestCase):
    def test_convert_weights_and_biases_to_decision_vector(self):
        weights = [np.array(((1, 2, 3), (4, 5, 6)))]
        biases = [np.array((7, 8, 9))]
        result = Network.convert_weights_and_biases_to_decision_vector(weights, biases)
        expected = np.array(range(1, 10), dtype=int)
        self.assertTrue(np.array_equal(expected, result))

    def test_weight_bias_conversion_reversible(self):
        weights = [np.random.rand(16, 784) * 2 - 1, np.random.rand(10, 16) * 2 - 1]
        biases = [np.random.rand(16) * 2 - 1, np.random.rand(10) * 2 - 1]
        v = Network.convert_weights_and_biases_to_decision_vector(weights, biases)
        new_weights, new_biases = Network.convert_decision_vector_to_weights_and_biases(v)
        for w1, w2 in zip(weights, new_weights):
            self.assertTrue(np.array_equal(w1, w2))
        for b1, b2 in zip(biases, new_biases):
            self.assertTrue(np.array_equal(b1, b2))
