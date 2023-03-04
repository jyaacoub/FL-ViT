"""
This file is for reading the data from the data folder

Data from: https://www.cs.toronto.edu/~kriz/cifar.html
"""
import pickle
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_data(file:str) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function reads the batch data from the file and returns 
    the images and labels.

    Args:
        file (str): path to the batch file
    
    returns:
        images (np.ndarray): (n_samples, 32, 32, 3)
        labels (np.ndarray): (n_samples,)
    """
    d = unpickle(file)
    imgs = d[b'data'] # (n_samples, 3072)
    labels = d[b'labels'] # (n_samples,)
    
    n = imgs.shape[0]
    
    imgs = imgs.reshape((n, 3, 32, 32))
    imgs = imgs.transpose((0, 2, 3, 1)) # (n, 32, 32, 3)
    
    return imgs, labels

if __name__ == "__main__":
    meta = unpickle('../data/cifar-10-batches-py/batches.meta')
    batch = unpickle('../data/cifar-10-batches-py/data_batch_1')

    data = lambda i: batch[b'data'][i]
    label = lambda i: meta[b'label_names'][batch[b'labels'][i]]

    sample_i = 4
    img = data(sample_i).reshape((3,32,32)).transpose((1,2,0))
    plt.imshow(img)
    plt.title(label(sample_i))
    plt.show()

