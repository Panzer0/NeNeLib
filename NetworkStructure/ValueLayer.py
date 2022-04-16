import numpy as np

import ActivationFunctions


class ValueLayer:
    def __init__(
        self,
        size,
        activationMethod=ActivationFunctions.noMethod,
        activationMethodDeriv=ActivationFunctions.noMethodDeriv,
    ):
        print(f"Size = {size}")
        self.values = np.zeros((1, size))
        self.delta = np.zeros((1, size))

        self.activationMethod = activationMethod
        self.activationMethodDeriv = activationMethodDeriv

    def getSize(self):
        return self.values.size

    def setMethod(
        self,
        activationMethod=ActivationFunctions.noMethod,
        activationMethodDeriv=ActivationFunctions.noMethodDeriv,
    ):
        self.activationMethod = activationMethod
        self.activationMethodDeriv = activationMethodDeriv

    def __str__(self):
        return str(self.values)

    def applyMethod(self):
        self.values = self.activationMethod(self.values)

    def getAfterDeriv(self):
        return [self.activationMethodDeriv(value) for value in self.values[0]]
