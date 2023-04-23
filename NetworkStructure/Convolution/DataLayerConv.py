import numpy as np

### PROCEDURE
#   Split input
#   Multiply by filters
from ActivationFunctions.ReLU import ReLU

ALPHA = 0.01

IMG_Y = 6
IMG_X = 5


####    Below is a skeleton of the entire CNN rather than just the data layer


def calc_kernel_layer(sections, kernels):
    return np.dot(sections, kernels.T)


class DataLayerConv:
    def __init__(self, expected, input=None):
        if input is None:
            self.image = np.random.randint(low=0, high=10, size=(IMG_Y, IMG_X))
        else:
            self.image = input
        self.expected = expected
        # todo: Should the two be merged? If I filters are just glorified
        #       weights, that'd make things much easier
        # self.weights = list()
        self.weights = list()

    def add_filter(self, new_filter=None):
        if new_filter is None:
            self.filters.append(np.random.uniform(low=0, high=10, size=(5, 6)))
        else:
            self.filters.append(new_filter)

    def add_weights(self, new_weights=None):
        if new_weights is None:
            self.weights.append(np.random.uniform(low=0, high=10, size=(5, 6)))
        else:
            self.weights.append(new_weights)

    def split(self, size, stride=1):
        subarrays = [
            self.image[y : y + size, x : x + size].T.flatten()
            for x in range(self.image.shape[1] - size + 1)[::stride]
            for y in range(self.image.shape[0] - size + 1)[::stride]
        ]
        return np.array(subarrays)

    def __str__(self):
        return str(self.image)
