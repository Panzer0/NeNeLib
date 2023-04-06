import tensorflow as tf
import numpy as np

MAX_TRAIN_SIZE = 60000
MAX_TEST_SIZE = 10000


class MNISTHandler:
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (
            (self.train_input, self.train_output),
            (self.test_input, self.test_output),
        ) = mnist.load_data()

    def get_adjusted_input(self, array):
        return array.reshape(array.shape[0], -1) / 255

    def get_train_input(self, amount=None):
        if amount and amount > MAX_TRAIN_SIZE:
            print(
                f"Insufficient data to fulfil request! {amount} > {MAX_TRAIN_SIZE}"
            )
            amount = MAX_TRAIN_SIZE
        adjusted_input = self.get_adjusted_input(self.train_input)
        return adjusted_input[:amount] if amount else adjusted_input

    def get_train_output(self, amount=None):
        if amount and amount > MAX_TRAIN_SIZE:
            print(
                f"Insufficient data to fulfil request! {amount} > {MAX_TRAIN_SIZE}"
            )
            amount = MAX_TRAIN_SIZE
        output = np.zeros((len(self.train_output), 1, 10))
        for i, label in enumerate(self.train_output):
            output[i][0][label] = 1
        return output[:amount] if amount else output

    def get_test_input(self, amount=None):
        if amount and amount > MAX_TEST_SIZE:
            print(
                f"Insufficient data to fulfil request! {amount} > {MAX_TEST_SIZE}"
            )
            amount = MAX_TEST_SIZE
        adjusted_input = self.get_adjusted_input(self.test_input)
        return adjusted_input[:amount] if amount else adjusted_input

    def get_test_output(self, amount=None):
        if amount and amount > MAX_TEST_SIZE:
            print(
                f"Insufficient data to fulfil request! {amount} > {MAX_TEST_SIZE}"
            )
            amount = MAX_TEST_SIZE
        output = np.zeros((len(self.test_output), 1, 10))
        for i, label in enumerate(self.test_output):
            output[i][0][label] = 1
        return output[:amount] if amount else output
