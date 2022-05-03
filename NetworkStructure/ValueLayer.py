import numpy as np

from ActivationFunctions.NoFunction import NoFunction


class ValueLayer:
    def __init__(self, size, activationFunction=NoFunction):
        print(f"Size = {size}")
        self.values = np.zeros((1, size))
        self.delta = np.zeros((1, size))

        self.activationFunction = activationFunction

    def getSize(self):
        return self.values.size

    def setMethod(self, activationFunction=NoFunction):
        self.activationFunction = activationFunction

    def __str__(self):
        return str(self.values)

    def applyMethod(self):
        self.values = self.activationFunction.function(self.values)

    def getAfterDeriv(self):
        return [
            self.activationFunction.derivative(value)
            for value in self.values[0]
        ]
