import numpy as np

from ActivationFunctions.NoFunction import NoFunction
from ActivationFunctions.ReLU import ReLU
from ActivationFunctions.Sigmoid import Sigmoid


class ValueLayer:
    def __init__(self, size, activationFunction=NoFunction, dropoutOdds=1.0):
        self.values = np.zeros((1, size))
        self.mask = np.ones((1, size))
        self.delta = np.zeros((1, size))
        self.activationFunction = activationFunction

        self.dropoutOdds = dropoutOdds

    def generateMask(self, probability: float) -> None:
        self.mask = np.random.rand(1, self.mask.size) < probability
        # res = np.multiply(weights, binary_value)
        # res /= probability
        # print(res)

    def applyDropout(self):
        self.applyMask()
        self.adjustForDropout()

    def applyDropoutNewMask(self):
        self.generateMask(self.dropoutOdds)
        self.applyMask()
        self.adjustForDropout()

    def applyMask(self):
        self.values = np.multiply(self.values, self.mask)

    def applyMaskToDelta(self):
       self.delta = np.multiply(self.delta, self.mask)

    def adjustForDropout(self):
        self.values = np.multiply(self.values, 1 / self.dropoutOdds)

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


# layer = ValueLayer(10, NoFunction, 0.5)
# layer.values[0][4] = 3
# layer.values[0][3] = -3
# layer.values[0][2] = 8
# layer.values[0][1] = 5
# print("Layer: " + str(layer))
# layer.applyMethod()
# print("Layer after activation method: " + str(layer))
# layer.applyDropout()
# print("Mask: " + str(layer.mask))
# print("Layer after dropout: " + str(layer))
