import numpy as np

### PROCEDURE
#   Split input
#   Multiply by filters
from ActivationFunctions.ReLU import ReLU

ALPHA = 0.01

####    Below is a skeleton of the entire CNN rather than just the data layer


def calc_kernel_layer(sections, kernels):
    return np.dot(sections, kernels.T)


class DataLayerConv:
    def __init__(self, expected, input=None):
        if input is None:
            self.image = np.random.randint(low=0, high=10, size=(5, 6))
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

    def split(self, size):
        subarrays = [
            self.image[y : y + size, x : x + size].T.flatten()
            for x in range(self.image.shape[1] - size + 1)
            for y in range(self.image.shape[0] - size + 1)
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

    kernel_layer = calc_kernel_layer(data.split(3), data.weights[0])
    kernel_layer = ReLU.function(kernel_layer)
    ## Entering final layer, flattening
    kernel_layer = kernel_layer.flatten()

    layer_2_values = calc_kernel_layer(kernel_layer, data.weights[1])
    # Network delta
    layer_2_delta = layer_2_values - data.expected
    ## Backpropagation
    layer_1_delta = np.dot(layer_2_delta, data.weights[1])
    layer_1_delta = layer_1_delta * ReLU.derivative(kernel_layer[np.newaxis, :])
    layer_1_delta = layer_1_delta.reshape(2, 2)

    # Very clunky np operations. Surely this can be streamlined somehow.
    layer_2_weight_delta = np.dot(
        layer_2_delta.reshape(-1, 1), kernel_layer[np.newaxis, :]
    )
    layer_1_weight_delta = np.dot(layer_1_delta.T, data.split(3))

    data.weights[1] = data.weights[1] - ALPHA * layer_2_weight_delta
    data.weights[0] = data.weights[0] - ALPHA * layer_1_weight_delta

    print(data.weights[1])
    print(data.weights[0])


