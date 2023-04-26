import numpy as np

from NetworkStructure.Convolution.Conv import Conv
from NetworkStructure.Convolution.Activation import Activation
from NetworkStructure.Convolution.FullyConnected import FullyConnected
from NetworkStructure.Convolution.Pool import Pool
from NetworkStructure.Convolution.ValueLayerConv import ValueLayerConv

ALPHA = 0.01
FILTER_COUNT = 2
FILTER_SIZE = 3
PADDING = False

EXPECTED = np.array(
    [
        [0, 1],
        [1, 0]
    ])
IMAGE = np.array(
    [
        [[8.5, 0.65, 1.2], [9.5, 0.8, 1.3], [9.9, 0.8, 0.5], [9.0, 0.9, 1.0]],
        [[7.6, 1.5, 4.9], [4.2, 4.9, 5.1], [8.9, 0.2, 0.3], [4.0, 0.4, 7.0]]
    ]
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
        self.layer_1 = ValueLayerConv()

        self.conv = Conv(
            filters=FILTERS, dim=FILTER_SIZE, filter_count=FILTER_COUNT
        )
        # self.pool = Pool()
        self.activation = Activation()
        # todo: Remember to remove WEIGHTS, this is meant to be random
        # todo: At the same time I have to remember to set the shape parameter
        # todo: Determined as such:
        # todo: h - Output parameter count
        # todo: w - layer_1 pixel count (flattened width)
        self.full_con = FullyConnected(WEIGHTS)

    def forward_propagate(self, input):
        self.layer_1.values = self.activation.apply(self.conv.apply(input))

        # todo: Pooling goes here

        # todo: It'd be cool if I could remove the flattening process and make
        # todo: it internal to FullyConnected, though it comes with its own set
        # todo: of issues.
        self.layer_1.values = self.layer_1.values.flatten()
        return self.full_con.apply(self.layer_1.values)

    def fit(self):
        for input, expected in zip(self.input, self.expected):
            # todo: Implement pooling, remove magic numbers
            out_delta = self.forward_propagate(input) - expected
            # print(f"For {expected}: {out_delta}")
            ## Layer 1 delta calculation
            # FC layer
            self.layer_1.delta = np.dot(out_delta, self.full_con.weights)
            # Activation function
            self.layer_1.delta = self.layer_1.delta * self.activation.apply_deriv(self.layer_1.values[np.newaxis, :])

            # Restoring original shape
            # todo: Hard coded right now, I'll make this dependent on pooling
            self.layer_1.delta = self.layer_1.delta.reshape(2, 2)

            self.full_con.back_propagate(out_delta, self.layer_1.values, ALPHA)
            self.conv.back_propagate(self.layer_1.delta, input, ALPHA)


if __name__ == "__main__":
    network = ConvNetwork(IMAGE, EXPECTED)
    for i in range(1):
        network.fit()
    # print(network.full_con.weights)
    # print(network.conv.filters)
