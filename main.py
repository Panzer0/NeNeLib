import pickle

import numpy as np

from Data import Data


class NeuralNetwork:
    def __init__(self, inputSize, firstLayerSize):
        self.values = list()
        self.weights = list()
        self.dataset = list()
        self.inputSize = inputSize
        self.outputSize = firstLayerSize

        self.weights.append(np.random.rand(firstLayerSize, inputSize))
        self.values.append(np.zeros(firstLayerSize))

        self.blankData()

    def isEmpty(self) -> bool:
        return len(self.weights) > 0

    def blankData(self):
        self.dataset.clear()
        # self.dataset.append(
        #     Data(np.ones((1, self.inputSize)), np.ones((1, self.outputSize))))

    def getOutputLayer(self):
        return self.values[-1]

    def display(self):
        for weight, values, enumerate in \
                zip(self.weights, self.values, range(len(self.weights))):
            print(f"{weight} w[{enumerate}] \n{values} v[{enumerate}]")

    def addLayer(self, size):
        # print(f"Size = {self.values[-1].size}")
        self.weights.append(np.random.rand(size, self.values[-1].size))
        self.values.append(np.zeros((1, size)))
        self.outputSize = size
        self.blankData()

    def addLayerRange(self, size, minValue, maxValue):
        difference = abs(minValue - maxValue)
        self.weights.append(
            difference * np.random.rand(size, self.values[-1].size) + minValue)
        self.values.append(np.zeros((1, size)))
        self.outputSize = size
        self.blankData()

    def refreshValues(self):
        self.values = list()
        for layer in self.weights:
            self.values.append(np.zeros((1, layer.shape[0])))

    def load(self, filename):
        with open(filename, 'rb') as handle:
            self.weights = pickle.load(handle)
        self.refreshValues()

    def save(self, filename):
        with open(filename, 'wb') as handle:
            pickle.dump(self.weights, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def predict(self, inputData):
        # Invalid input handling
        if inputData.size != self.inputSize:
            print(
                f"Invalid input data size, {inputData.size} != {self.inputSize}")
            return

        self.values[0] = inputData.dot(
            self.weights[0].T)  # Multiplying the input data
        for i in range(1, len(self.values)):
            self.values[i] = self.values[i - 1].dot(self.weights[i].T)
        return self.values[-1]

    def fit(self):
        for sample in self.dataset:
            output = self.predict(sample.input)
            delta = output - sample.output
            print(f"delta = {delta}")

            # todo: Replace weight with a reference
            # for weight in self.weights[::-1]:
            #     wDelta = delta @ weight
            #     weight = weight - 0.05 * wDelta
            #     delta = delta @ weight

            # todo: Have another look at the weights used, something's not right. Weight[0] is discarded.
            for n in range(len(self.values) - 1, -1, -1):
                if n > 0:
                    wDelta = delta @ self.weights[n]
                else:
                    wDelta = delta.T @ sample.input
                delta = delta @ self.weights[n]
                self.weights[n] = \
                    self.weights[n] - 0.05 * wDelta

    def updateLatestDataManual(self):
        for i in range(len(network.dataset[-1].input[0])):
            self.dataset[-1].input[0][i] = float(input("Enter input value"))
        print(self.dataset[-1].input[0])

        for i in range(len(self.dataset[-1].output[0])):
            self.dataset[-1].output[0][i] = float(input("Enter output value"))
        print(self.dataset[-1].output[0])

    def addSampleManual(self):
        network.dataset.append(
            Data(np.ones((1, network.inputSize)), np.ones((1, network.outputSize))))
        network.updateLatestDataManual()

    def displayDataset(self):
        for data in network.dataset:
            print(f"Input:{data.input[0]}")
            print(f"Output:{data.output[0]}\n")

    def addSampleColour(self, r: float, g: float, b: float, colour: int):
        network.dataset.append(
            Data(np.zeros((1, network.inputSize)), np.zeros((1, network.outputSize))))

        self.dataset[-1].input[0][0] = r
        self.dataset[-1].input[0][1] = g
        self.dataset[-1].input[0][2] = b

        print(f"colour = {colour}")
        self.dataset[-1].output[0][colour-1] = 1

        print(f"Appending input{self.dataset[-1].input[0]}, output = {self.dataset[-1].output[0]}")

    def loadColourFile(self, filename):
        with open(filename, 'r') as handle:
            data = [*map(float, handle.read().split())]

        flag = 0
        for i in range(len(data)):
            if flag == 0:
                print(f"0, {data[i]}")
                r = data[i]
                flag += 1
            elif flag == 1:
                print(f"1, {data[i]}")
                g = data[i]
                flag += 1
            elif flag == 2:
                print(f"2, {data[i]}")
                b = data[i]
                flag += 1
            elif flag == 3:
                print(f"3, {data[i]}")
                out = data[i]
                self.addSampleColour(r, g, b, int(out))
                flag = 0

    def validateColours(self):
        total = 0
        correct = 0
        for sample in self.dataset:
            total += 1
            result = self.predict(sample.input)
            print(result)
            print(sample.output[0])
            print(f"Argmax = {np.argmax(result[0])}")

            if np.argmax(result[0]) == np.argmax(sample.output[0]):
                print("Correct!")
                correct += 1
            else:
                print(f"WRONG!, {np.argmax(result[0])} != {np.argmax(sample.output[0])}")
        return float(correct/total * 100)




'''
    def fit(self, input, expected):
        # Invalid input handling
        if input.size != self.inputSize:
            print("Invalid input data size")
            return
        if expected.size != self.getOutputLayer().size:
            print("Invalid expected data size")
            return

        output = self.predict(input)  # Calculating the output layer's values
        output_delta = output - expected  # Calculating the output layer's delta
        print(f"Output:\n{output}")
        print(f"Output delta = {output_delta}")

        # Adjusting values
        for i in range(len(self.weights) - 1, 0, -1):
            print(f"i = {i}")
            if i == len(self.weights) - 1:
                delta = output_delta
            else:
                delta = np.dot(delta.T, self.weights[i])  # Calculating the delta of the lower layer neurons
            if i == 0:
                weighted_delta = np.outer(delta, input)                # Calculating the final layer's delta
            else:
                weighted_delta = np.outer(delta, self.values[i-1])  # Calculating the given layer's delta
            self.weights[i] = self.weights[i] - 0.02 * weighted_delta        # Adjusting the weights
'''

# todo: File handling
# todo: More advanced user features
#   Replace np.ones data with custom values
# todo: Fix fit() for multi-layer nets


inputData = np.ones((1, int(input("Enter input data size:"))))
firstLayerSize = int(input("Enter first layer size"))
network = NeuralNetwork(inputData.size, firstLayerSize)
while True:
    print("0 - Add quick layer\n"
          "1 - Add custom layer\n"
          "2 - Fit\n"
          "3 - Display\n"
          "4 - Predict\n"
          "5 - Save\n"
          "6 - Load\n"
          "7 - Overwrite latest data\n"
          "8 - Append new data\n")
    operation = int(input("Choose operation:"))
    if operation == 0:
        network.addLayer(int(input("Enter layer size")))
    if operation == 1:
        network.addLayerRange(
            int(input("Enter layer size")),
            int(input("Enter min weight value")),
            int(input("Enter max weight value")))
    if operation == 2:
        for i in range(50):
            network.fit()
            print("\n")
    if operation == 3:
        network.display()
    if operation == 4:
        print(network.predict(network.dataset[0].input))
    if operation == 5:
        network.save("data.pickle")
    if operation == 6:
        network.load("data.pickle")
    if operation == 7:
        # Read data, temporary solution
        network.updateLatestDataManual()
        network.displayDataset()
    if operation == 8:
        network.addSampleManual()
        network.displayDataset()
    if operation == 9:
        network.loadColourFile("colours.txt")
    if operation == 10:
        print(f"{network.validateColours()}%")


# inputData = np.ones(3)
# expectedData = np.ones(1)+4
# print(f"Expected data: {expectedData}")

# network = NeuralNetwork(inputData.size, 4)

# network.addLayer(2)
# network.addLayer(3)
# network.addLayer(1)

# network.display()
# network.fit(inputData, expectedData)

# print(np.outer(np.ones(3), np.ones(3)))
