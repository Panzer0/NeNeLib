import numpy as np

### PROCEDURE
#   Split input
#   Multiply by filters
from ActivationFunctions.ReLU import ReLU

ALPHA = 0.01

IMG_Y = 6
IMG_X = 5


####    Below is a skeleton of the entire CNN rather than just the data layer


def calc_kernel_layer(sections, kernels):
    return np.dot(sections, kernels.T)


class DataLayerConv:
    def __init__(self, expected, input=None):
        if input is None:
            self.image = np.random.randint(low=0, high=10, size=(IMG_Y, IMG_X))
        else:
            self.image = input
        self.expected = expected
        # todo: Should the two be merged? If I filters are just glorified
        #       weights, that'd make things much easier
        # self.weights = list()
        self.weights = list()

    def add_filter(self, new_filter=None):
        if new_filter is None:
            self.filters.append(np.random.uniform(low=0, high=10, size=(5, 6)))
        else:
            self.filters.append(new_filter)

    def add_weights(self, new_weights=None):
        if new_weights is None:
            self.weights.append(np.random.uniform(low=0, high=10, size=(5, 6)))
        else:
            self.weights.append(new_weights)

    def split(self, size, stride=1):
        subarrays = [
            self.image[y: y + size, x: x + size].T.flatten()
            for x in range(self.image.shape[1] - size + 1)[::stride]
            for y in range(self.image.shape[0] - size + 1)[::stride]
        ]
        return np.array(subarrays)

    def __str__(self):
        return str(self.image)


if __name__ == "__main__":
    expected = np.array([0, 1])
    image = np.array(
        [[8.5, 0.65, 1.2], [9.5, 0.8, 1.3], [9.9, 0.8, 0.5], [9.0, 0.9, 1.0]]
    )
    filters = np.array(
        [
            [0.1, 0.2, -0.1, -0.1, 0.1, 0.9, 0.1, 0.4, 0.1],
            [0.3, 1.1, -0.3, 0.1, 0.2, 0.0, 0.0, 1.3, 0.1],
        ]
    )
    weights = np.array([[0.1, -0.2, 0.1, 0.3], [0.2, 0.1, 0.5, -0.3]])

    data = DataLayerConv(expected, image)
    data.add_weights(filters)
    data.add_weights(weights)


    layer_1_values = np.dot(data.split(3), data.weights[0].T)
    layer_1_values = ReLU.function(layer_1_values)
    print(layer_1_values)
    ## Exiting final convolution layer, flattening
    #### NOTE:  The row count corresponds to the amount of image slices, while
    ####        the column count corresponds to the amount of filters

    #todo: Could the only difference really be the need to reshape the data
    #todo: when leaving/entering the convolution layer? But what about pooling?
    layer_1_values = layer_1_values.flatten()

    layer_2_values = np.dot(layer_1_values, data.weights[1].T)
    # Network delta
    layer_2_delta = layer_2_values - data.expected
    ## Backpropagation
    layer_1_delta = np.dot(layer_2_delta, data.weights[1])
    layer_1_delta = layer_1_delta * ReLU.derivative(layer_1_values[np.newaxis, :])
    layer_1_delta = layer_1_delta.reshape(2, 2)

    # Very clunky np operations. Surely this can be streamlined somehow.
    layer_2_weight_delta = np.dot(
        layer_2_delta.reshape(-1, 1), layer_1_values[np.newaxis, :]
    )
    layer_1_weight_delta = np.dot(layer_1_delta.T, data.split(3))

    data.weights[1] = data.weights[1] - ALPHA * layer_2_weight_delta
    data.weights[0] = data.weights[0] - ALPHA * layer_1_weight_delta

    print(data.weights[1])
    print(data.weights[0])
