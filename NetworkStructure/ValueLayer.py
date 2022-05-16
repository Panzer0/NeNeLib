import numpy as np

import ActivationFunctions.Sigmoid
from ActivationFunctions.NoFunction import NoFunction
from ActivationFunctions.ReLU import ReLU
from ActivationFunctions.Sigmoid import Sigmoid


class ValueLayer:
    def __init__(self, size, activationFunction=NoFunction):
        print(f"Size = {size}")
        self.values = np.zeros((1, size))
        self.mask = np.ones((1, size))
        self.delta = np.zeros((1, size))

        self.activationFunction = activationFunction

    def generateMask(self, probability: float) -> None:
        self.mask = np.random.rand(1, self.mask.size) < probability
        # res = np.multiply(weights, binary_value)
        # res /= probability
        # print(res)

    def getMasked(self):
        return np.multiply(self.values, self.mask)

    def getSize(self):
        return self.values.size

    def setMethod(self, activationFunction=NoFunction):
        self.activationFunction = activationFunction

    def __str__(self):
        return str(self.values)

    def applyMethod(self):
        self.values = self.activationFunction.function(self.values)

    def getAfterDeriv(self):
        return self.activationFunction.derivative(self.values)


#
# layer = ValueLayer(10, NoFunction)
# layer.values[0][4] = 3
# layer.values[0][3] = -3
# layer.values[0][2] = 8
# layer.values[0][1] = 5
# print(layer)
# layer.applyMethod()
# print(layer)
# print(layer.getAfterDeriv())
