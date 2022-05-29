from math import exp

import numpy as np


class Sigmoid:
    @staticmethod
    def function(layer):
        return np.array(
            [[1 / (1 + exp(-value)) for value in batch] for batch in layer]
        )

    @staticmethod
    def derivative(layer):
        return [[value * (1 - value) for value in batch] for batch in layer]
