from data_loader import load_data
from network import Network


if __name__ == '__main__':
    data_x, data_y = load_data(download=True)
    network = Network()
    try:
        network.train(data_x, data_y)
    except KeyboardInterrupt:
        pass
    finally:
        network.save()
