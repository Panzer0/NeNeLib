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
    def __init__(self, function=ReLU):
        self.function = function

    def apply(self, image):
        return self.function.function(image)

    # todo
    def back_propagate(self, data):
        return self.function.derivative(data)


if __name__ == '__main__':
    conv = Conv()
    print(conv)
