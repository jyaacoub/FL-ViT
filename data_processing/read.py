"""
This file is for reading the data from the data folder

Data from: https://www.cs.toronto.edu/~kriz/cifar.html
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def display_sample(sample: np.ndarray):
    """
    Each row of the array stores a 32x32 colour image. 
        The first 1024 entries contain the red channel values, 
        the next 1024 the green, and 
        the final 1024 the blue. 
    
    The image is stored in row-major order, so that the 
    first 32 entries of the array are the red channel 
    values of the first row of the image.

    Args:
        sample (np.ndarray): (3072,) array of the sample
    """
    img = sample.reshape((3,32,32)).transpose((1,2,0))
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    d = unpickle('../data/cifar-10-batches-py/batches.meta')
    d1 = unpickle('../data/cifar-10-batches-py/data_batch_1')

    data = lambda i: d1[b'data'][i]
    label = lambda i: d[b'label_names'][d1[b'labels'][i]]

    display_sample(data(1))
    print(label(0))
