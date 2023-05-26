import numpy as np

from MNISTHandler import MNISTHandler
from NetworkStructure.Convolution.Conv import Conv
from NetworkStructure.Convolution.Activation import Activation
from NetworkStructure.Convolution.FullyConnected import FullyConnected
from NetworkStructure.Convolution.Pool import Pool
from NetworkStructure.Convolution.ValueLayerConv import ValueLayerConv

ALPHA = 0.01
FILTER_COUNT = 16
FILTER_SIZE = 3
GENERATION_COUNT = 300
PADDING = False


class ConvNetwork:
    def __init__(self, input, expected):
        # todo: Create a new Data class that unites input and expected
        self.input = input
        self.expected = expected
        self.layer_1 = ValueLayerConv()

        self.conv = Conv(dim=FILTER_SIZE, filter_count=FILTER_COUNT)
        self.pool = Pool()
        self.activation = Activation()

        # Calculating the shape of the weights in the FC layer
        ## h - Output parameter count
        ## w - Amount of values in the preceding layer
        self.layer_1_shape = self.pool.calc_output_dims(
            self.conv.calc_output_dims(input[0].shape)
        )
        weight_h = self.expected.shape[1]
        weight_w = self.layer_1_shape[0] * self.layer_1_shape[1]
        self.full_con = FullyConnected(shape=(weight_h, weight_w))

    def forward_propagate(self, input):
        self.layer_1.values = self.activation.apply(self.conv.apply(input))
        self.layer_1.pooled_values = self.pool.apply(self.layer_1.values)
        self.layer_1.pooled_values = self.layer_1.pooled_values.flatten()
        return self.full_con.apply(self.layer_1.pooled_values)

    def fit(self):
        for input, expected in zip(self.input, self.expected):
            out_delta = self.forward_propagate(input) - expected
            # print(f"For {expected}: {out_delta + expected}")
            # FC layer
            self.layer_1.delta = np.dot(out_delta, self.full_con.weights)
            # Restoring original shape
            self.layer_1.delta = self.layer_1.delta.reshape(self.layer_1_shape)
            # Pool layer
            self.layer_1.delta = self.pool.expand_deltas(self.layer_1.delta)
            # Activation function
            self.layer_1.delta = (
                self.layer_1.delta
                * self.activation.apply_deriv(self.layer_1.values)
            )

            self.full_con.back_propagate(
                out_delta, self.layer_1.pooled_values, ALPHA
            )
            self.conv.back_propagate(self.layer_1.delta, input, ALPHA)
            # print("|", end="")
        # print("-")

    def validate_multi_class(self, input_data, output_data):
        total, correct = 0, 0
        for input, output in zip(input_data, output_data):
            total += 1
            result = self.forward_propagate(input)
            # print(f"For {np.argmax(output)}: {np.argmax(result)} (second:{np.argsort(result)[-2]})")
            correct += np.argmax(result) == np.argmax(output)
        return float(correct / total * 100)


if __name__ == "__main__":
    handler = MNISTHandler()
    train_input = handler.get_train_input(amount=1_000, conv=True)
    train_output = handler.get_train_output(amount=1_000, conv=True)
    test_input = handler.get_test_input(amount=1_000, conv=True)
    test_output = handler.get_test_output(amount=1_000, conv=True)

    network = ConvNetwork(train_input, train_output)
    # network = ConvNetwork(IMAGE, EXPECTED)
    for i in range(GENERATION_COUNT):
        network.fit()
        print(
            f"For testing  [{i}]: {network.validate_multi_class(test_input, test_output)}"
        )
        print(
            f"For training [{i}]: {network.validate_multi_class(train_input, train_output)}\n"
        )
