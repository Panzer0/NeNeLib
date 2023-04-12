import numpy as np
from ActivationFunctions.NoFunction import NoFunction


class ValueLayerBatch:
    def __init__(
        self,
        batchSize,
        layerSize,
        activationFunction=NoFunction,
        dropoutOdds=1.0,
    ):
        self.values = np.zeros((batchSize, layerSize))
        self.mask = np.ones((batchSize, layerSize))
        self.delta = np.zeros((batchSize, layerSize))
        self.activationFunction = activationFunction

        self.batchSize = batchSize
        self.dropoutOdds = dropoutOdds