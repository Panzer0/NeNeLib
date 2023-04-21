import numpy as np

from NetworkStructure.Convolution.Conv import Conv
from NetworkStructure.Convolution.Activation import Activation

ALPHA = 0.01
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
        self.conv = Conv(filters = FILTERS, dim=FILTER_SIZE, filter_count=FILTER_COUNT)
        # todo
        # self.pool = Pool()
        self.activation = Activation()
        # todo
        # self.fully_connected = FullyConnected()
        # todo: Temp replacement ----v
        self.weights = WEIGHTS

    #todo: Making layer_1_values a self value is terribly clunky. Change it.
    # todo: This is where I should use the ValueLayer type. Vals and deltas.
    def forward_propagate(self, input):
        self.layer_1_values = self.conv.apply(input)
        self.layer_1_values = self.activation.apply(self.layer_1_values)
        # todo: Pooling goes here
        self.layer_1_values = self.layer_1_values.flatten()
        # todo: Replace the below with a FullyConnected object
        return np.dot(self.layer_1_values, self.weights.T)

    def fit(self):
        for input, expected in zip(self.input, self.expected):
            # todo: The below is pretty much a copy-paste of that clunky main
            # todo: method. Magic numbers have to be replaced, code has to be
            # todo: streamlined.
            output_delta = self.forward_propagate(input) - expected
            print(f"Output delta is \n----------\n{output_delta}\n----------")

            layer_1_delta = np.dot(output_delta, self.weights)
            layer_1_delta = layer_1_delta * self.activation.back_propagate(self.layer_1_values[np.newaxis, :])
            layer_1_delta = layer_1_delta.reshape(2, 2)

            weight_delta = np.dot(
                output_delta.reshape(-1, 1), self.layer_1_values[np.newaxis, :]
            )
            filters_delta = np.dot(layer_1_delta.T, self.conv.split(input, 3))

            self.weights = self.weights - ALPHA * weight_delta
            self.conv.filters = self.conv.filters - ALPHA * filters_delta








if __name__ == '__main__':
    network = ConvNetwork(IMAGE, EXPECTED)
    for i in range(10):
        network.fit()