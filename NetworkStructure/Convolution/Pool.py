import numpy as np


class Pool:
    def __init__(self, size=2, stride=2):
        self.size = size
        self.stride = stride

    def calc_h(self, image) -> int:
        return int((image.shape[0] - self.size) / self.stride + 1)

    def calc_w(self, image) -> int:
        return int((image.shape[1] - self.size) / self.stride + 1)

    def calc_dims(self, image):
        return self.calc_h(image), self.calc_w(image)

    def pool(self, image):
        h, w = self.calc_dims(image)
        subarrays = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                patch = image[
                        i * self.stride: i * self.stride + self.size,
                        j * self.stride: j * self.stride + self.size
                        ]
                subarrays[i, j] = np.max(patch)
        return subarrays

    def get_max_indices(self, image):
        h, w = self.calc_dims(image)
        indices = np.zeros((h, w, 2), dtype=np.int32)
        for i in range(h):
            for j in range(w):
                patch = image[
                        i * self.stride: i * self.stride + self.size,
                        j * self.stride: j * self.stride + self.size
                        ]
                index = np.unravel_index(np.argmax(patch), patch.shape)
                indices[i, j] = (
                    i * self.stride + index[0], j * self.stride + index[1])
        return indices

    def get_mask(self, image):
        indices = self.get_max_indices(image)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[indices[..., 0], indices[..., 1]] = 1
        return mask

    def apply(self, image):
        self.mask = self.get_mask(image)
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
    print(image)
    print(pll.mask)
