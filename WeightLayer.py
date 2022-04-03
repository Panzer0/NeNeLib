import ActivationFunctions


class WeightLayer:
    def __init__(self, weights, activationMethod=None, activationMethodDeriv = None):
        self.weights = weights
        if activationMethod is None:
            self.activationMethod = ActivationFunctions.noMethod
            self.activationMethodDeriv = ActivationFunctions.noMethodDeriv
        self.activationMethod = activationMethod
        self.activationMethodDeriv = activationMethodDeriv

    def getShape(self):
        return self.weights.shape

    def __str__(self):
        return str(self.weights)
