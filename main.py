from numpy.random import randint

from data_loader import load_data
from plot import show_img
from network import Network


if __name__ == '__main__':
    i = randint(500)
    data_x, data_y = load_data()
    print("Expected: ", data_y[i])
    network = Network()
    result = network.find_value(data_x[i])
    print("Result: ", result)
    show_img(data_x[i])
