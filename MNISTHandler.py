import tensorflow
import numpy as np

MAX_SIZE = 10000


class MNISTHandler:

    def __init__(self):
        mnist = tensorflow.keras.datasets.mnist
        (
            (self.train_input, self.train_output),
            (self.test_input, self.test_output),
        ) = mnist.load_data()

    def getAdjustedInput(self, array):
        out = [np.array([x.flatten()]) for x in array]
        return np.array([x / 255 for x in out])

    def getTrainInput(self, amount=MAX_SIZE):
        return self.getAdjustedInput(self.train_input)[0:amount]

    def getTrainOutput(self, amount=MAX_SIZE):
        output = np.zeros((len(self.train_output), 1, 10))
        for i, label in enumerate(self.train_output):
            output[i][0][label] = 1
        return output[0:amount]

    def getTestInput(self, amount=MAX_SIZE):
        return self.getAdjustedInput(self.test_input)[0:amount]

    def getTestOutput(self, amount=MAX_SIZE):
        output = np.zeros((len(self.test_output), 1, 10))
        for i, label in enumerate(self.test_output):
            output[i][0][label] = 1
        return output[0:amount]
