import numpy as np

from ActivationFunctions.NoFunction import NoFunction
from ActivationFunctions.ReLU import ReLU
from ActivationFunctions.Sigmoid import Sigmoid


class ValueLayerBatch:
    def __init__(self, batchSize, layerSize, activationFunction=NoFunction,
                 dropoutOdds=1.0):
        self.values = np.zeros((batchSize, layerSize))
        self.mask = np.ones((batchSize, layerSize))
        self.delta = np.zeros((batchSize, layerSize))
        self.activationFunction = activationFunction

        self.batchSize = batchSize
        self.dropoutOdds = dropoutOdds

    def generateMask(self, probability: float) -> None:
        self.mask = np.random.rand(self.batchSize, self.getSize()) < probability

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
        return self.values.shape[1]

    def setMethod(self, activationFunction=NoFunction):
        self.activationFunction = activationFunction

    def __str__(self):
        return str(self.values)

    def applyMethod(self):
        self.values = self.activationFunction.function(self.values)

    def getAfterDeriv(self):
        return self.activationFunction.derivative(self.values)


# layer = ValueLayerBatch(2, 5, ReLU, 0.5)
# layer.values[0][0] = 5
# layer.values[0][1] = 5
# layer.values[0][2] = 8
# layer.values[0][3] = -3
# layer.values[0][4] = 3
# layer.values[1][0] = 65
# layer.values[1][1] = -9
# layer.values[1][2] = 0
# layer.values[1][3] = 999
# layer.values[1][4] = 0
# print("Layer: \n" + str(layer))
# layer.applyMethod()
# print("Layer after activation method: \n" + str(layer))
# layer.applyDropoutNewMask()
# print("Mask: \n" + str(layer.mask))
# print("Layer after dropout: \n" + str(layer))
