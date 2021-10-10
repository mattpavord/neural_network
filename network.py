import pickle
from copy import deepcopy
import random

import numpy as np

import utils


class Network:
    """
    Main class for neural network logic
    Ref - http://neuralnetworksanddeeplearning.com/chap2.html
    """

    training_sample_size = 5000
    neuron_sizes = [784, 16, 10]
    step_size = 3

    def __init__(self):
        self.weights, self.biases = self.get_empty_weight_bias_shapes()
        try:
            self.load()
        except (FileNotFoundError, EOFError):
            print("Could not find existing memory, starting from scratch")
            self.weights = [np.random.rand(16, 784) * 2 - 1, np.random.rand(10, 16) * 2 - 1]
            self.biases = [np.zeros(16), np.zeros(10)]

    @classmethod
    def get_empty_weight_bias_shapes(cls):
        """ Get empty data structures of shapes for weights and biases """
        weights = [np.zeros((cls.neuron_sizes[i+1], cls.neuron_sizes[i])) for i in range(len(cls.neuron_sizes) - 1)]
        biases = [np.zeros(cls.neuron_sizes[i+1]) for i in range(len(cls.neuron_sizes) - 1)]
        return weights, biases

    def save(self):
        data = [self.weights, self.biases]
        pickle.dump(data, open("decision_data.pickle", "wb"))

    def load(self):
        with open('decision_data.pickle', 'rb') as handle:
            data = pickle.load(handle)
        self.weights, self.biases = data

    def get_output_vector(self, img):
        """
        Return the output of the neural network in a 1d array of numbers
        :param img: 2D image array
        :return:
        """
        v = utils.convert_2d_to_vector(img) / 256
        for weight_matrix, bias_vector in zip(self.weights, self.biases):
            v = utils.sigmoid(np.matmul(weight_matrix, v) + bias_vector)
        return v

    def find_value(self, img):
        output = self.get_output_vector(img)
        return np.argmax(output)

    def cost_function(self, img_data, expected_data):
        cost_elements = []
        for img, expected in zip(img_data, expected_data):
            expected_vector = np.zeros(10)
            expected_vector[expected] = 1
            output_vector = self.get_output_vector(img)
            cost_elements.append(np.linalg.norm(output_vector - expected_vector) ** 2)
        return sum(cost_elements) / (2 * self.training_sample_size)

    def feed_forward(self, img_vector: np.array):
        """
        Feed forward - from image vector calculate all activations and normalised activations
        :param img_vector: img reshaped to a 1D array of image
        :return:
            a_vectors (list(np.array) normalised activation, a = sigmoid(z)
            z_vectors (list(np.array) - activations for hidden layers and final layer
        """
        a_vectors = [img_vector]
        z_vectors = [None]  # no need to store z values of input vector
        for weight_matrix, bias_vector in zip(self.weights, self.biases):
            z_vector = np.matmul(weight_matrix, a_vectors[-1]) + bias_vector
            z_vectors.append(z_vector)
            a_vectors.append(utils.sigmoid(z_vector))
        return a_vectors, z_vectors

    def backpropagate_error_vectors(self, expected_vector: np.array, a_vectors, z_vectors):
        """
        Error vectors are the way we calculate the gradient of the cost function with respect to
        the weights or the bias elements
        They are defined as the gradient of the cost function with respect to z (activation before normalisation)

        :param expected_vector - len 10 vector of expected result, e.g. for a digit of 1 this will be (0, 1, 0, 0, ...)
        :param a_vectors - list of activation vectors (output of feed_forward)
        :param z_vectors - list of normalised activation vectors (output of feed_forward)
        :return: List of error vectors -
            Note that despite being a backpropagation algorithm, the list will remain in order of layers,
            i.e. the error vector of the first hidden layer will be the first element of the list
        """
        grad_cost_a = a_vectors[-1] - expected_vector  # gradient of cost function with respect to activation
        error_vectors = [grad_cost_a * utils.sigmoid_prime(z_vectors[-1])]
        for i in range(1, len(self.neuron_sizes) - 1):
            error_vector = np.matmul(self.weights[-i].transpose(), error_vectors[-i]) * utils.sigmoid_prime(z_vectors[-i-1])
            error_vectors.insert(0, error_vector)
        return error_vectors

    def train(self, img_data, expected_data, n_iterations=1000):
        """
        Adjusts weights using a gradient descent method
        img_data: Array of 2D image vectors
        expected_data: Array of values
        n_iterations(int): Number of gradient descent iterations
        """
        prev_n_correct = []
        for _ in range(n_iterations):
            test_indexes = random.sample(range(len(img_data)), self.training_sample_size)
            img_sample = img_data[test_indexes]
            expected_sample = expected_data[test_indexes]

            n_correct = 0
            delta_weights, delta_biases = self.get_empty_weight_bias_shapes()
            for img, expected in zip(img_sample, expected_sample):
                img_v = utils.convert_2d_to_vector(img) / 256  # normalise values to between 0 and 1
                expected_vector = utils.get_expected_vector(expected)
                a_vectors, z_vectors = self.feed_forward(img_v)
                error_vectors = self.backpropagate_error_vectors(expected_vector, a_vectors, z_vectors)
                for i in range(len(error_vectors)):
                    delta_weights[i] += np.outer(error_vectors[i], a_vectors[i]) / self.training_sample_size
                    delta_biases[i] += error_vectors[i] / self.training_sample_size
                    # divide by self.training_sample_size so that we're iteratively calculating
                    # the average across all training tests
                n_correct += 1 if np.argmax(a_vectors[-1]) == expected else 0

            for i in range(len(self.weights)):
                self.weights[i] -= delta_weights[i] * self.step_size
                self.biases[i] -= delta_biases[i] * self.step_size

            if len(prev_n_correct) > 5:
                prev_n_correct.pop(0)

            if len(prev_n_correct) > 3 and all([prev_n_correct[i-1] > prev_n_correct[i]
                                                for i in range(1, len(prev_n_correct))]):
                print("Reducing step size")
                self.step_size /= 2
            prev_n_correct.append(n_correct)

            print(_, '\t', n_correct, "/", self.training_sample_size, '\t', self.step_size)



