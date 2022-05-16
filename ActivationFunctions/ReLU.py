import numpy as np


class ReLU:
    @staticmethod
    def function(x):
        return np.maximum(0, x)

    @staticmethod
    def derivative(layer):
        return [[int(value > 0) for value in layer[0]]]
