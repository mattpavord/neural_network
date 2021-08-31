import pickle


def load_data(download=False, use_test_data=False):
    if download:
        from keras.datasets import mnist
        (train_x, train_y), (test_x, test_y) = mnist.load_data()
        print("Finished download")
        if use_test_data:
            return test_x, test_y
    else:
        with open('train_x.pickle', 'rb') as handle:
            train_x = pickle.load(handle)
        with open('train_y.pickle', 'rb') as handle:
            train_y = pickle.load(handle)
    return train_x, train_y
