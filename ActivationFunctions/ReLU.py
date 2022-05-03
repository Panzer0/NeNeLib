import numpy as np


class ReLU:
    @staticmethod
    def function(x):
        return np.maximum(0, x)

    @staticmethod
    def derivative(x):
        return 1 if x > 0 else 0
