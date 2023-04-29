import tensorflow as tf
import numpy as np

MAX_TRAIN_SIZE = 60000
MAX_TEST_SIZE = 10000


def get_adjusted_input(array):
    return array.reshape(array.shape[0], -1) / 255


def get_conv_input(array):
    return array / 255


def apply_amount_cap(amount, max_size):
    if amount and amount > max_size:
        print(
            f"Insufficient data to fulfil request! {amount} > {max_size}"
        )
        amount = max_size
    return amount


class MNISTHandler:
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (
            (self.train_input, self.train_output),
            (self.test_input, self.test_output),
        ) = mnist.load_data()

    def get_train_input(self, amount=None, conv=False):
        amount = apply_amount_cap(amount, MAX_TRAIN_SIZE)
        adjusted_input = get_conv_input(
            self.train_input) if conv else get_adjusted_input(self.train_input)
        return adjusted_input[:amount] if amount else adjusted_input

    def get_train_output(self, amount=None, conv=False):
        amount = apply_amount_cap(amount, MAX_TRAIN_SIZE)
        if conv:
            output = np.zeros((len(self.train_output), 10))
            for i, label in enumerate(self.train_output):
                output[i][label] = 1
        else:
            output = np.zeros((len(self.train_output), 1, 10))
            for i, label in enumerate(self.train_output):
                output[i][0][label] = 1
        return output[:amount] if amount else output

    def get_test_input(self, amount=None, conv=False):
        amount = apply_amount_cap(amount, MAX_TEST_SIZE)
        adjusted_input = get_conv_input(
            self.test_input) if conv else get_adjusted_input(self.test_input)
        return adjusted_input[:amount] if amount else adjusted_input

    def get_test_output(self, amount=None, conv=False):
        amount = apply_amount_cap(amount, MAX_TEST_SIZE)
        if conv:
            output = np.zeros((len(self.test_output), 10))
            for i, label in enumerate(self.test_output):
                output[i][label] = 1
        else:
            output = np.zeros((len(self.test_output), 1, 10))
            for i, label in enumerate(self.test_output):
                output[i][0][label] = 1
        return output[:amount] if amount else output
