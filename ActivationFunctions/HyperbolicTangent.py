from math import exp

import numpy as np


class HyperbolicTangent:
    @staticmethod
    def function(batch):
        return np.array(
            [
                [
                    (exp(value) - exp(-value)) / (exp(value) + exp(-value))
                    for value in layer
                ]
                for layer in batch
            ]
        )

    @staticmethod
    def derivative(batch):
        return [[1 - pow(value, 2) for value in layer] for layer in batch]
