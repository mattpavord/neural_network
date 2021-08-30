from matplotlib import pyplot as plt


def show_img(img):
    """
    Plot image
    :param img: 2D numpy array
    """
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.show()
