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
        return self.calc_output_h(img_shape[0]), self.calc_output_w(
            img_shape[1]
        )

    def pool(self, image):
        """
        :return:
            max_patches: The max-pooled image
            mask: A mask which indicates the selected values
        """
        h, w = self.calc_output_dims(image.shape)
        # h x w patches, each self.size x self.size shaped
        patches = image.reshape(h, self.size, w, self.size)
        max_patches = np.max(patches, axis=(1, 3))
        mask = (patches == max_patches[:, np.newaxis, :, np.newaxis])
        mask = mask.reshape(h * self.size, w * self.size)
        return max_patches, mask

    def inflate_deltas(self, deltas):
        return np.repeat(
            np.repeat(deltas, self.size, axis=0), self.size, axis=1
        )

    def expand_deltas(self, deltas):
        return self.mask * self.inflate_deltas(deltas)

    def apply(self, image):
        pooled, self.mask = self.pool(image)
        return pooled


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
    deltas = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    pll = Pool()
    print(pll.apply(image))
    print(pll.mask)
    print(f"\n-----\n{pll.expand_deltas(deltas)}\n-----")
