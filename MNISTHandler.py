import tensorflow
import numpy as np


class MNISTHandler:
    def __init__(self):
        mnist = tensorflow.keras.datasets.mnist
        (
            (self.train_input, self.train_output),
            (self.test_input, self.test_output),
        ) = mnist.load_data()

    def adjustInput(self, array):
        return [np.array([x.flatten()]) for x in array]

    def getTrainInput(self):
        return self.adjustInput(self.train_input)

    def getTrainOutput(self):
        output = np.zeros((len(self.train_output), 1, 10))
        for i, label in enumerate(self.train_output):
            output[i][0][label] = 1
        return output

    def getTestInput(self):
        return self.adjustInput(self.test_input)

    def getTestOutput(self):
        output = np.zeros((len(self.test_output), 10))
        for i, label in enumerate(self.test_output):
            output[i][label] = 1
        return output


