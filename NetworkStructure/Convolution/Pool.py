import numpy as np


class Pool:
    def __init__(self, size=2, stride=2):
        self.size = size
        self.stride = stride

    def calc_output_h(self, image_h) -> int:
        return int((image_h - self.size) / self.stride + 1)

    def calc_output_w(self, image_w) -> int:
        return int((image_w - self.size) / self.stride + 1)

    def calc_output_dims(self, img_shape):
        return self.calc_output_h(img_shape[0]), self.calc_output_w(img_shape[1])

    def pool(self, image):
        h, w = self.calc_output_dims(image.shape)
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
        h, w = self.calc_output_dims(image.shape)
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

    def inflate_deltas(self, deltas):
        return np.repeat(np.repeat(deltas, self.size, axis=0), self.size, axis=1)

    def expand_deltas(self, deltas):
        return self.mask * self.inflate_deltas(deltas)

    def apply(self, image):
        self.mask = self.get_mask(image)
        return self.pool(image)




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
    deltas = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
    )
    pll = Pool()
    print(pll.apply(image))
    print(pll.mask)
    print(f"\n-----\n{pll.expand_deltas(deltas)}\n-----")
