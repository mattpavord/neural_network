from copy import deepcopy
import random

import numpy as np

import utils


class Network:
    training_sample_size = 20
    neuron_sizes = [784, 16, 10]

    def __init__(self):
        self.weights = [np.random.rand(16, 784) * 2 - 1, np.random.rand(10, 16) * 2 - 1]
        self.biases = [np.zeros(16), np.zeros(10)]

    def get_output_vector(self, img):
        v = utils.convert_2d_to_vector(img) / 256
        for weight_matrix, bias_vector in zip(self.weights, self.biases):
            v = utils.sigmoid(np.matmul(weight_matrix, v) + bias_vector)
        return v

    @staticmethod
    def convert_weights_and_biases_to_decision_vector(weights, biases):
        """
        Decision vector is just a 1D representation of all degrees of freedom
        I.e. All elements of weight matrices and bias vectors
        Organised so that all weight elements appear first in order, followed by biases
        """
        vector_size = 0
        for m in [*weights, *biases]:
            vector_size += m.size
        decision_vector = np.zeros(vector_size)
        starting_index = 0
        for m in [*weights, *biases]:
            decision_vector[starting_index: starting_index + m.size] = m.flatten()
            starting_index += m.size
        return decision_vector

    @classmethod
    def convert_decision_vector_to_weights_and_biases(cls, decision_vector):
        n_elements = len(cls.neuron_sizes) - 1
        weights = [np.zeros((cls.neuron_sizes[i+1], cls.neuron_sizes[i])) for i in range(n_elements)]
        biases = [np.zeros(cls.neuron_sizes[i+1]) for i in range(n_elements)]
        index = 0
        for i in range(n_elements):  # weight matrices
            rows, columns = weights[i].shape
            for j in range(rows):
                weights[i][j] = decision_vector[index: index+columns]
                index += columns
        for i in range(n_elements):  # bias vectors
            biases[i] = decision_vector[index: index+len(biases[i])]
            index += len(biases[i])
        return weights, biases

    def find_value(self, img):
        output = self.get_output_vector(img)
        return np.argmax(output)

    def cost(self, img_data, expected_data):
        cost_elements = []
        for img, expected in zip(img_data, expected_data):
            expected_vector = np.zeros(10)
            expected_vector[expected] = 1
            output_vector = self.get_output_vector(img)
            cost_elements.append(np.linalg.norm(output_vector - expected_vector) ** 2)
        return sum(cost_elements) / (2 * self.training_sample_size)

    def train(self, img_data, expected_data, n_iterations=100):
        """
        Adjusts weights using a gradient descent method
        img_data: Array of 2D image vectors
        expected_data: Array of values
        n_iterations(int): Number of gradient descent iterations
        """
        original_weights = deepcopy(self.weights)
        original_biases = deepcopy(self.biases)
        for _ in range(n_iterations):
            test_indexes = random.sample(range(len(img_data)), self.training_sample_size)
            cost = self.cost(img_data[test_indexes], expected_data[test_indexes])
