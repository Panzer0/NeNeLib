import numpy as np

WEIGHTS_LOW = -0.01
WEIGHTS_HIGH = 0.01


class FullyConnected:
    def __init__(self, weights=None, shape=(2, 4)):
        if weights is None:
            # todo: Test the randomly generated variant
            self.weights = np.random.uniform(
                low=-WEIGHTS_LOW, high=WEIGHTS_HIGH, size=shape
            )
        else:
            self.weights = weights

    def apply(self, image):
        return np.dot(image.flatten(), self.weights.T)

    # todo
    def back_propagate(self, delta, values, alpha):
        weight_delta = np.dot(
            delta.reshape(-1, 1), values[np.newaxis, :]
        )
        self.weights = self.weights - alpha * weight_delta


if __name__ == "__main__":
    image = np.array(
        [[8.5, 0.65, 1.2], [9.5, 0.8, 1.3], [9.9, 0.8, 0.5], [9.0, 0.9, 1.0]]
    )
    filters = np.array(
        [
            [0.1, 0.2, -0.1, -0.1, 0.1, 0.9, 0.1, 0.4, 0.1],
            [0.3, 1.1, -0.3, 0.1, 0.2, 0.0, 0.0, 1.3, 0.1],
        ]
    )
    conv = Conv(filters)

    print(conv.apply(image))
