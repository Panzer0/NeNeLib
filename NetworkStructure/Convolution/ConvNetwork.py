import numpy as np

from NetworkStructure.Convolution.Conv import Conv
from NetworkStructure.Convolution.Activation import Activation
from NetworkStructure.Convolution.FullyConnected import FullyConnected
from NetworkStructure.Convolution.Pool import Pool
from NetworkStructure.Convolution.ValueLayerConv import ValueLayerConv

ALPHA = 0.001
FILTER_COUNT = 3
FILTER_SIZE = 3
PADDING = False

EXPECTED = np.array(
    [
        [0, 1, 0],
        [1, 0, 0]
    ])
IMAGE = np.array(
    [
        [
            [3, 6, 7, 5],
            [6, 2, 9, 1],
            [0, 9, 3, 6],
            [2, 6, 1, 8],
        ],
        [
            [1, 6, 7, 5],
            [1, 2, 9, 1],
            [1, 9, 3, 6],
            [1, 6, 1, 8],
        ]
    ]
)
FILTERS = np.array(
    [
        [0.1, 0.2, -0.1, -0.1, 0.1, 0.9, 0.1, 0.4, 0.1],
        [0.3, 1.1, -0.3, 0.1, 0.2, 0.0, 0.0, 1.3, 0.1],
        [-0.6, 1.8, -0.1, 0.7, 0.5, 0.7, -0.5, 3.3, 0.1],
        [0.2, -0.6, -0.6, 0.9, 0.4, -0.6, -0.3, 2.1, 1.0],
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
        self.pool = Pool()
        self.activation = Activation()
        # todo: Remember to remove WEIGHTS, this is meant to be random
        # todo: At the same time I have to remember to set the shape parameter
        # todo: Determined as such:
        # todo: h - Output parameter count
        # todo: w - layer_1 pixel count (flattened width)
        self.layer_1_shape = self.pool.calc_output_dims(self.conv.calc_output_dims(input[0].shape))
        print(self.layer_1_shape)
        weight_w = self.layer_1_shape[0] * self.layer_1_shape[1]
        weight_h = self.expected.shape[1]
        print(weight_h, weight_w)
        self.full_con = FullyConnected(shape=(weight_h, weight_w))

    def forward_propagate(self, input):
        self.layer_1.values = self.activation.apply(self.conv.apply(input))

        # todo: Pooling goes here
        self.layer_1.pooled_values = self.pool.apply(self.layer_1.values)

        # todo: It'd be cool if I could remove the flattening process and make
        # todo: it internal to FullyConnected, though it comes with its own set
        # todo: of issues.
        # print("input\n", input)
        # print("filters\n", self.conv.filters)
        # print("values\n", self.layer_1.values)
        # print("pooled_values\n", self.layer_1.pooled_values)
        # print("weights\n", self.full_con.weights)
        self.layer_1.pooled_values = self.layer_1.pooled_values.flatten()
        return self.full_con.apply(self.layer_1.pooled_values)

    def fit(self):
        for input, expected in zip(self.input, self.expected):
            # print("##################PACKPROPAGATION##################")
            # todo: Implement pooling, remove magic numbers
            out_delta = self.forward_propagate(input) - expected
            print(f"For {expected}: {out_delta}")
            ## Layer 1 delta calculation
            # FC layer
            self.layer_1.delta = np.dot(out_delta, self.full_con.weights)

            # Restoring original shape
            # todo: Hard coded right now, I'll make this dependent on pooling
            self.layer_1.delta = self.layer_1.delta.reshape(self.layer_1_shape)

            # Pool layer
            self.layer_1.delta = self.pool.expand_deltas(self.layer_1.delta)
            # print("layer_1.delta expanded\n", self.layer_1.delta)

            # Activation function
            self.layer_1.delta = self.layer_1.delta * self.activation.apply_deriv(self.layer_1.values)
            # print("layer_1.delta activated\n", self.layer_1.delta)
            #
            # print("layer_1.values\n", self.layer_1.values)

            self.full_con.back_propagate(out_delta, self.layer_1.pooled_values, ALPHA)
            self.conv.back_propagate(self.layer_1.delta, input, ALPHA)


if __name__ == "__main__":
    network = ConvNetwork(IMAGE, EXPECTED)
    for i in range(5000):
        network.fit()
    # print(network.full_con.weights)
    # print(network.conv.filters)
