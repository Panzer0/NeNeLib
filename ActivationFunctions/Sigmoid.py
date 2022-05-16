from math import exp

import numpy as np


class Sigmoid:
    @staticmethod
    def function(layer):
        return np.array([[1 / (1 + exp(-value)) for value in layer[0]]])

    @staticmethod
    def derivative(layer):
        return [[value * (1 - value) for value in layer[0]]]
