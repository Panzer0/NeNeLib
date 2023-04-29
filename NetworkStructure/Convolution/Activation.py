import numpy as np

from ActivationFunctions.ReLU import ReLU


def calc_kernel_layer(sections, kernels):
    return np.dot(sections, kernels.T)


class Activation:
    def __init__(self, function=ReLU):
        self.function = function

    def apply(self, image):
        return self.function.function(image)

    def apply_deriv(self, data):
        return self.function.derivative(data)


if __name__ == "__main__":
    pass
