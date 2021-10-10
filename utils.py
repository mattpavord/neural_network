import numpy as np


def convert_2d_to_vector(img):
    """ Convert 2d array to 1d """
    return np.reshape(img, img.shape[0] * img.shape[1])


def get_expected_vector(value: int):
    expected_vector = np.zeros(10, dtype=int)
    expected_vector[value] = 1
    return expected_vector


def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z))


def sigmoid_prime(z):
    """Derivative of sigmoid function"""
    return sigmoid(z) * (1 - sigmoid(z))
