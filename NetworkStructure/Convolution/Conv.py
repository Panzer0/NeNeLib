from math import sqrt
import numpy as np

LOW = -0.01
HIGH = 0.01


class Conv:
    def __init__(
            self, filters=None, filter_count=1, dim=3, stride=1, padding=False
    ):
        if filters is None:
            self.filters = np.random.uniform(
                low=LOW, high=HIGH, size=(filter_count, dim * dim)
            )
        else:
            self.filters = filters
        self.f_shape = int(sqrt(self.filters.shape[1]))
        self.stride = stride
        self.padding = padding

    def calc_output_dims(self, img_shape):
        temp_img = np.zeros(img_shape)
        h = self.split(temp_img).shape[0]
        w = self.filters.shape[0]
        return h, w

    def split(self, image):
        img_copy = np.pad(image, ((1, 1), (1, 1))) if self.padding else image
        subarrays = [
            img_copy[y: y + self.f_shape, x: x + self.f_shape].T.flatten()
            for x in range(img_copy.shape[1] - self.f_shape + 1)[:: self.stride]
            for y in range(img_copy.shape[0] - self.f_shape + 1)[:: self.stride]
        ]
        return np.array(subarrays)

    def apply(self, image):
        split_image = self.split(image)
        return np.dot(split_image, self.filters.T)

    def back_propagate(self, delta, values, alpha):
        filters_delta = np.dot(delta.T, self.split(values))
        self.filters = self.filters - alpha * filters_delta

    def __str__(self):
        return str(self.filters)


if __name__ == "__main__":
    image = np.array(
        [
            [8.5, 0.65, 1.2, 0.1],
            [9.5, 0.8, 1.3, 0.1],
            [9.9, 0.8, 0.5, 0.1],
            [9.0, 0.9, 1.0, 0.1],
            [9.9, 0.8, 0.5, 0.1],
        ]
    )
    filters = np.array(
        [
            [0.1, 0.2, -0.1, -0.1, 0.1, 0.9, 0.1, 0.4, 0.1],
            [0.3, 1.1, -0.3, 0.1, 0.2, 0.0, 0.0, 1.3, 0.1],
            [0.3, 1.1, -0.3, 0.1, 0.2, 0.0, 0.0, 1.3, 0.1],
        ]
    )
    conv = Conv(filters)

    print(conv.apply(image))
    print(conv.calc_output_dims(image.shape))
