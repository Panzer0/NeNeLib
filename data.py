from numpy import ndarray


class Data:
    def __init__(self, network_input: ndarray, network_output: ndarray):
        self.input = network_input
        self.output = network_output

    def matchesSize(self, other_data):
        return self.input == other_data.input \
               and self.output == other_data.output

    def __str__(self):
        return "Input: " + str(self.input) + "\nOutput: " + str(self.output)
