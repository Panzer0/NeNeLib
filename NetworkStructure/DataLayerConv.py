import numpy
import numpy as np


class DataLayerConv:
    def __init__(self):
        self.image = np.random.randint(low=0, high=10, size=(5, 6))

    def split(self, size):
        subarrays = [self.image[y:y + size, x:x + size].flatten()
                     for y in range(self.image.shape[0] - size + 1)
                     for x in range(self.image.shape[1] - size + 1)]
        return np.array(subarrays)

    def __str__(self):
        return str(self.image)


if __name__ == "__main__":
    data = DataLayerConv()
    print(data)
    print(data.split(3))
