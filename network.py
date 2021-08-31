import pickle
from copy import deepcopy
import random

import numpy as np

import utils


class Network:
    training_sample_size = 20
    neuron_sizes = [784, 16, 10]
    step_size = 0.4

    def __init__(self):
        self.weights = [np.random.rand(16, 784) * 2 - 1, np.random.rand(10, 16) * 2 - 1]
        self.biases = [np.zeros(16), np.zeros(10)]
        try:
            self.load()
        except (FileNotFoundError, EOFError):
            print("Could not find existing memory, starting from scratch")

    def save(self):
        data = [self.weights, self.biases]
        pickle.dump(data, open("decision_data.pickle", "wb"))

    def load(self):
        with open('decision_data.pickle', 'rb') as handle:
            data = pickle.load(handle)
        self.weights, self.biases = data

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
        """
        Inverse of convert_weights_and_biases_to_decision_vector
        I.e. Unpack a 1D "decision_vector" into the weight matrices and bias vectors
        """
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

    def get_output_vector(self, img, decision_vector=None):
        """
        Return the output of the neural network in a 1d array of numbers
        :param img: 2D image array
        :param decision_vector: Optional[1D Array] to set weights and biases - if None then use self.weights, self.biases
        :return:
        """
        if decision_vector is None:
            weights = self.weights
            biases = self.biases
        else:
            weights, biases = self.convert_decision_vector_to_weights_and_biases(decision_vector)
        v = utils.convert_2d_to_vector(img) / 256
        for weight_matrix, bias_vector in zip(weights, biases):
            v = utils.sigmoid(np.matmul(weight_matrix, v) + bias_vector)
        return v

    def find_value(self, img, decision_vector=None):
        output = self.get_output_vector(img, decision_vector)
        return np.argmax(output)

    def cost_function(self, img_data, expected_data, decision_vector=None):
        cost_elements = []
        for img, expected in zip(img_data, expected_data):
            expected_vector = np.zeros(10)
            expected_vector[expected] = 1
            output_vector = self.get_output_vector(img, decision_vector)
            cost_elements.append(np.linalg.norm(output_vector - expected_vector) ** 2)
        return sum(cost_elements) / (2 * self.training_sample_size)

    def train(self, img_data, expected_data, n_iterations=100):
        """
        Adjusts weights using a gradient descent method
        img_data: Array of 2D image vectors
        expected_data: Array of values
        n_iterations(int): Number of gradient descent iterations
        """
        decision_vector = self.convert_weights_and_biases_to_decision_vector(self.weights, self.biases)
        dx = 0.001

        for _ in range(n_iterations):
            test_indexes = random.sample(range(len(img_data)), self.training_sample_size)
            img_sample = img_data[test_indexes]
            expected_sample = expected_data[test_indexes]
            cost = self.cost_function(img_sample, expected_sample)

            grad_cost = np.zeros(len(decision_vector))  # gradient vector for cost function
            for i in range(len(decision_vector)):
                aug_decision_vector = deepcopy(decision_vector)
                aug_decision_vector[i] += dx
                aug_cost = self.cost_function(img_sample, expected_sample, aug_decision_vector)
                grad_cost[i] = (aug_cost - cost) / dx

            decision_vector -= self.step_size * grad_cost
            self.weights, self.biases = self.convert_decision_vector_to_weights_and_biases(decision_vector)
            new_cost = self.cost_function(img_sample, expected_sample)
            if new_cost > cost:  # this shouldn't happen -> step size is too big
                self.step_size /= 2
            print(_, '\t', round(cost, 6), '\t', round(new_cost, 6))


