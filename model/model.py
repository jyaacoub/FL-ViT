from data_processing.read import get_data, unpickle
from config import DATA_PATH_FN, DATA_PATH, LABEL_NAMES

import matplotlib.pyplot as plt

imgs, labels = get_data(DATA_PATH_FN(1)) # get the first batch

for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(imgs[i])
    plt.title(LABEL_NAMES[labels[i]])
    
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
