import numpy as np
from numpy import ndarray, squeeze


class DataBatch:
    def __init__(self, network_input: ndarray, network_output: ndarray, singular_batch=False):
        self.input = squeeze(network_input)
        self.output = squeeze(network_output)

        # The batch dimension is removed by np.squeeze if batch size is 1
        if singular_batch:
            self.input = self.input[np.newaxis, :]
            self.output = self.output[np.newaxis, :]

    def matchesSize(self, other_data):
        return (
            self.input == other_data.input and self.output == other_data.output
        )

    def __str__(self):
        return f"Input: {str(self.input)} \nOutput: {str(self.output)}"
