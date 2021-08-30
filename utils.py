import numpy as np


def convert_2d_to_vector(img):
    """ Convert 2d array to 1d """
    return np.reshape(img, img.shape[0] * img.shape[1])


def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z))
