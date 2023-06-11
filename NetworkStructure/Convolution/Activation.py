import numpy as np

from ActivationFunctions.ReLU import ReLU


class Activation:
    def __init__(self, function=ReLU):
        self.function = function

    def apply(self, image):
        return self.function.function(image)

    def apply_deriv(self, data):
        return self.function.derivative(data)
