import numpy as np

import ActivationFunctions


class ValueLayer:
    def __init__(self, size, activationMethod=None, activationMethodDeriv=None):
        print(f"Size = {size}")
        self.values = np.zeros((1, size))
        self.delta = np.zeros((1, size))

        if activationMethod is None:
            self.activationMethod = ActivationFunctions.noMethod
            self.activationMethodDeriv = ActivationFunctions.noMethodDeriv
        else:
            self.activationMethod = activationMethod
            self.activationMethodDeriv = activationMethodDeriv

    def getShape(self):
        return self.values.shape

    def getSize(self):
        return self.values.size

    def __str__(self):
        return str(self.values)
