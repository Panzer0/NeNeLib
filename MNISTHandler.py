import tensorflow


class MNISTHandler:
    def __init__(self):
        mnist = tensorflow.keras.datasets.mnist
        (
            (self.train_input, self.train_output),
            (self.test_input, self.test_output),
        ) = mnist.load_data()

    def adjustInput(self, array):
        return [x.flatten() for x in array]

    def getTrainInput(self):
        return self.adjustInput(self.train_input)

    def getTrainOutput(self):
        # todo: Adjust for neural network output
        return self.train_output

    def getTestInput(self):
        return self.adjustInput(self.train_input)

    def getTestOutput(self):
        # todo: Adjust for neural network output
        return self.train_output


handler = MNISTHandler()
input = handler.getTrainInput()
print(input[10])
