import numpy as np


def sum_tuples(first, second):
    return tuple(a + b for a, b in zip(first, second))


class Pool:
    def __init__(self, size=2, stride=2):
        self.size = size
        self.stride = stride

    def calc_h(self, image) -> int:
        return int((image.shape[0] - self.size) / self.stride + 1)

    def calc_w(self, image) -> int:
        return int((image.shape[1] - self.size) / self.stride + 1)

    def pool(self, image):
        subarrays = [
            max(max(row) for row in image[y : y + self.size, x : x + self.size])
            for y in range(image.shape[0] - self.size + 1)[:: self.stride]
            for x in range(image.shape[1] - self.size + 1)[:: self.stride]
        ]
        return np.array(subarrays).reshape(
            self.calc_h(image), self.calc_w(image)
        )

    def get_max_indices(self, image):
        indices = [
            sum_tuples(
                np.unravel_index(
                    (np.argmax(image[y : y + self.size, x : x + self.size])),
                    (self.size, self.size),
                ),
                (y, x),
            )
            for y in range(image.shape[0] - self.size + 1)[:: self.stride]
            for x in range(image.shape[1] - self.size + 1)[:: self.stride]
        ]
        return np.array(indices).reshape(
            self.calc_h(image), self.calc_w(image), 2
        )

    def apply(self, image):
        self.max_indices = self.get_max_indices(image)
        return self.pool(image)

    # todo
    def back_propagate(self, data):
        pass


if __name__ == "__main__":
    image = np.array(
        [
            [3, 6, 7, 5, 3, 5],
            [6, 2, 9, 1, 2, 7],
            [0, 9, 3, 6, 0, 6],
            [2, 6, 1, 8, 7, 9],
            [2, 0, 2, 3, 7, 5],
            [9, 2, 2, 8, 9, 7],
        ]
    )
    pll = Pool()
    print(pll.apply(image))
    print(pll.max_indices)
