import numpy as np

from NetworkStructure.Convolution.Conv import Conv
from NetworkStructure.Convolution.Activation import Activation
from NetworkStructure.Convolution.FullyConnected import FullyConnected
from NetworkStructure.Convolution.Pool import Pool
from NetworkStructure.Convolution.ValueLayerConv import ValueLayerConv

ALPHA = 0.01
FILTER_COUNT = 4
FILTER_SIZE = 3
GENERATION_COUNT = 500
PADDING = False

EXPECTED = np.array([[0, 1, 0], [1, 0, 0]])
IMAGE = np.array(
    [
        [
            [1, 0.7, 0.7, 0.5],
            [6, 0.2, 0.9, 1],
            [0, 0.9, 0.3, 0.6],
            [0.2, 0.6, 0.1, 0.8],
        ],
        [
            [1, 0.6, 0.7, 0.5],
            [0.1, 0.2, 0.9, 0.1],
            [1, 0.9, 0.3, 0.6],
            [0.1, 0.6, 1, 0.8],
        ],
    ]
)


# FILTERS = np.array(
#     [
#         [0.1, 0.2, -0.1, -0.1, 0.1, 0.9, 0.1, 0.4, 0.1],
#         [0.3, 1.1, -0.3, 0.1, 0.2, 0.0, 0.0, 1.3, 0.1],
#         [-0.6, 1.8, -0.1, 0.7, 0.5, 0.7, -0.5, 3.3, 0.1],
#         [0.2, -0.6, -0.6, 0.9, 0.4, -0.6, -0.3, 2.1, 1.0],
#     ]
# )
# WEIGHTS = np.array([[0.1, -0.2, 0.1, 0.3], [0.2, 0.1, 0.5, -0.3]])


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
            print(f"For {expected}:\t{out_delta}")
            # Layer 1 delta calculation
            ## FC layer
            self.layer_1.delta = np.dot(out_delta, self.full_con.weights)
            ## Restoring original shape
            self.layer_1.delta = self.layer_1.delta.reshape(self.layer_1_shape)
            ## Pool layer
            self.layer_1.delta = self.pool.expand_deltas(self.layer_1.delta)
            ## Activation function
            self.layer_1.delta = (
                    self.layer_1.delta
                    * self.activation.apply_deriv(self.layer_1.values)
            )

            self.full_con.back_propagate(
                out_delta, self.layer_1.pooled_values, ALPHA
            )
            self.conv.back_propagate(self.layer_1.delta, input, ALPHA)


if __name__ == "__main__":
    network = ConvNetwork(IMAGE, EXPECTED)
    for i in range(GENERATION_COUNT):
        network.fit()
