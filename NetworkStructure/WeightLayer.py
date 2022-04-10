import ActivationFunctions

#todo: Remove Activation method from uses of weightLayer()
class WeightLayer:
    def __init__(self, weights):
        self.weights = weights

    def getShape(self):
        return self.weights.shape

    def __str__(self):
        return str(self.weights)
