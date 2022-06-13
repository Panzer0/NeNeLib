from math import exp

import numpy as np


class SoftMax:
    @staticmethod
    def function(batch):
        sumList = [sum([exp(value) for value in layer]) for layer in batch]
        return np.array(
            [
                [exp(value) / layerSum for value in layer]
                for layer, layerSum in zip(batch, sumList)
            ]
        )

    @staticmethod
    def derivative(batch):
        return [[[1 - pow(value, 2) for value in layer] for layer in batch]]
