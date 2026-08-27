import os

import numpy as np
from torchvision.datasets import CIFAR10


def load_unified(root=None):
    #We set root as the location of our CIFAR10. 
    root = root or os.environ["FL_DATA_ROOT"]

    #We take each dataset
    train = CIFAR10(root=root, train=True, download=True)
    test = CIFAR10(root=root, train=False, download=True)

    #We combine them as specified
    x = np.concatenate([train.data, test.data], axis=0)
    y = np.concatenate([np.asarray(train.targets), np.asarray(test.targets)])

    print("Datasets unified")

    return x, y, train.classes
