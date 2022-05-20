import tensorflow
import numpy as np

MAX_TRAIN_SIZE = 60000
MAX_TEST_SIZE = 10000


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

    def getTrainInput(self, amount=None):
        if amount and amount > MAX_TRAIN_SIZE:
            print(f"Insufficient data to fulfil request! {amount} > {MAX_TRAIN_SIZE}")
            amount = MAX_TRAIN_SIZE
        retVal = self.getAdjustedInput(self.train_input)
        return retVal[0:amount] if amount else retVal

    def getTrainOutput(self, amount=None):
        if amount and amount > MAX_TRAIN_SIZE:
            print(f"Insufficient data to fulfil request! {amount} > {MAX_TRAIN_SIZE}")
            amount = MAX_TRAIN_SIZE
        output = np.zeros((len(self.train_output), 1, 10))
        for i, label in enumerate(self.train_output):
            output[i][0][label] = 1
        return output[0:amount] if amount else output

    def getTestInput(self, amount=None):
        if amount and amount > MAX_TEST_SIZE:
            print(f"Insufficient data to fulfil request! {amount} > {MAX_TEST_SIZE}")
            amount = MAX_TEST_SIZE
        retval = self.getAdjustedInput(self.test_input)
        return retval[0:amount] if amount else retval

    def getTestOutput(self, amount=None):
        if amount and amount > MAX_TEST_SIZE:
            print(f"Insufficient data to fulfil request! {amount} > {MAX_TEST_SIZE}")
            amount = MAX_TEST_SIZE
        output = np.zeros((len(self.test_output), 1, 10))
        for i, label in enumerate(self.test_output):
            output[i][0][label] = 1
        return output[0:amount] if amount else output
