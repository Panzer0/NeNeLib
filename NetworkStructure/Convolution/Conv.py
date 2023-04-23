from math import sqrt
import numpy as np


def split(image, size, stride=1, padding=False):
    image_copy = np.pad(image, ((1, 1), (1, 1))) if padding else image
    subarrays = [
        image_copy[y: y + size, x: x + size].T.flatten()
        for x in range(image_copy.shape[1] - size + 1)[::stride]
        for y in range(image_copy.shape[0] - size + 1)[::stride]
    ]
    return np.array(subarrays)


class Conv:
    def __init__(self, filters=None, filter_count=1, dim=3, stride=1):
        if filters is None:
            self.filters = np.random.uniform(
                low=-0.01, high=0.01, size=(filter_count, dim * dim)
            )
        else:
            self.filters = filters
        # todo: Should the two be merged? If I filters are just glorified
        #       weights, that'd make things much easier
        # self.weights = list()
        self.filter_dim = int(sqrt(self.filters.shape[1]))
        self.stride = stride

    def apply(self, image):
        split_image = split(image, self.filter_dim, self.stride)
        return np.dot(split_image, self.filters.T)

    def back_propagate(self, delta, values, alpha):
        filters_delta = np.dot(
            delta.T, split(values, self.filter_dim, self.stride)
        )
        self.filters = self.filters - alpha * filters_delta

    def __str__(self):
        return str(self.filters)


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
