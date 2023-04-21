import numpy as np

from NetworkStructure.Convolution.Conv import Conv
from NetworkStructure.Convolution.Activation import Activation

FILTER_COUNT = 2
FILTER_SIZE = 3
PADDING = False

EXPECTED = np.array([[0, 1]])
IMAGE = np.array(
    [[[8.5, 0.65, 1.2], [9.5, 0.8, 1.3], [9.9, 0.8, 0.5], [9.0, 0.9, 1.0]]]
)
FILTERS = np.array(
    [
        [0.1, 0.2, -0.1, -0.1, 0.1, 0.9, 0.1, 0.4, 0.1],
        [0.3, 1.1, -0.3, 0.1, 0.2, 0.0, 0.0, 1.3, 0.1],
    ]
)
WEIGHTS = np.array([[0.1, -0.2, 0.1, 0.3], [0.2, 0.1, 0.5, -0.3]])


class ConvNetwork:
    def __init__(self, input, expected):
        # todo: Create a new Data class that unites input and expected
        self.input = input
        self.expected = expected
        print(self.input.shape)
        self.conv = Conv(filters = FILTERS, dim=FILTER_SIZE, filter_count=FILTER_COUNT)
        # todo
        # self.pool = Pool()
        self.activation = Activation()
        # todo
        # self.fully_connected = FullyConnected()
        # todo: Temp replacement ----v
        self.weights = WEIGHTS

    def forward_propagate(self, input):
        layer_1_values = self.conv.apply(input)
        layer_1_values = self.activation.apply(layer_1_values)
        # todo: Pooling goes here
        layer_1_values = layer_1_values.flatten()
        # todo: Replace the below with a FullyConnected object
        return np.dot(layer_1_values, self.weights.T)

    def fit(self):
        for input, expected in zip(self.input, self.expected):
            print(f"input is {input}")
            output_delta = self.forward_propagate(input) - expected
            print(output_delta)



            # layer_2_delta = layer_2_values - data.expected
            # ## Backpropagation
            # layer_1_delta = np.dot(layer_2_delta, data.weights[1])
            # layer_1_delta = layer_1_delta * ReLU.derivative(
            #     layer_1_values[np.newaxis, :])
            # layer_1_delta = layer_1_delta.reshape(2, 2)
            #
            # # Very clunky np operations. Surely this can be streamlined somehow.
            # layer_2_weight_delta = np.dot(
            #     layer_2_delta.reshape(-1, 1), layer_1_values[np.newaxis, :]
            # )
            # layer_1_weight_delta = np.dot(layer_1_delta.T, data.split(3))
            #
            # data.weights[1] = data.weights[1] - ALPHA * layer_2_weight_delta
            # data.weights[0] = data.weights[0] - ALPHA * layer_1_weight_delta





if __name__ == '__main__':
    network = ConvNetwork(IMAGE, EXPECTED)
    network.fit()