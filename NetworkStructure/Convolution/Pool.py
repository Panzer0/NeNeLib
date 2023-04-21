from math import sqrt

import numpy as np

### PROCEDURE
#   Split input
#   Multiply by filters
from ActivationFunctions.ReLU import ReLU

dim = 3


####    Below is a skeleton of the entire CNN rather than just the data layer


def calc_kernel_layer(sections, kernels):
    return np.dot(sections, kernels.T)


class Conv:
    def __init__(self, filters=None, stride=1):
        if filters is None:
            self.filters = np.random.uniform(low=-0.5, high=0.5, size=(1, dim*dim))
        else:
            self.filters = filters
        # todo: Should the two be merged? If I filters are just glorified
        #       weights, that'd make things much easier
        # self.weights = list()
        self.weights = list()
        self.filter_dim = int(sqrt(self.filters.shape[1]))
        self.stride = stride
        print(self.filter_dim)

    def split(self, image, size, stride=1):
        subarrays = [
            image[y: y + size, x: x + size].T.flatten()
            for x in range(image.shape[1] - size + 1)[::stride]
            for y in range(image.shape[0] - size + 1)[::stride]
        ]
        return np.array(subarrays)

    def apply(self, image):
        split_image = self.split(image, self.filter_dim, self.stride)
        return np.dot(split_image, self.filters)

    # todo
    def back_propagate(self, data):
        pass


    def __str__(self):
        return str(self.filters)


if __name__ == '__main__':
    conv = Conv()
    print(conv)
