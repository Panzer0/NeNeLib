import pickle

import numpy as np

import ActivationFunctions
from NetworkStructure.Data import Data
from NetworkStructure.ValueLayer import ValueLayer
from NetworkStructure.WeightLayer import WeightLayer

ALPHA = 0.01


class NeuralNetwork:
    def __init__(self, inputSize, firstLayerSize):
        self.values = list()
        self.weightLayers = list()
        self.dataset = list()
        self.inputSize = inputSize
        self.outputSize = firstLayerSize

        self.weightLayers.append(
            WeightLayer(np.random.rand(firstLayerSize, inputSize))
        )
        self.values.append(
            ValueLayer(
                firstLayerSize,
                ActivationFunctions.ReLU,
                ActivationFunctions.ReLUDeriv,
            )
        )

        self.blankData()

    def isEmpty(self) -> bool:
        return len(self.weightLayers) > 0

    def blankData(self):
        self.dataset.clear()
        # self.dataset.append(
        #     Data(np.ones((1, self.inputSize)), np.ones((1, self.outputSize))))

    def getOutputLayer(self):
        return self.values[-1]

    def display(self):
        for weight, values, index in zip(
            self.weightLayers, self.values, range(len(self.weightLayers))
        ):
            print(f"{weight} w[{index}]\n" f"{values} v[{index}]")

    def addLayer(self, size):
        # print(f"Size = {self.values[-1].size}")
        self.weightLayers.append(
            WeightLayer(np.random.rand(size, self.values[-1].getSize()))
        )
        self.values.append(
            ValueLayer(
                size, ActivationFunctions.ReLU, ActivationFunctions.ReLUDeriv
            )
        )
        self.outputSize = size
        self.blankData()

    def addLayerRange(self, size, minValue, maxValue):
        difference = abs(minValue - maxValue)
        self.weightLayers.append(
            WeightLayer(
                difference * np.random.rand(size, self.values[-1].getSize())
                + minValue
            )
        )
        self.values.append(
            ValueLayer(
                size, ActivationFunctions.ReLU, ActivationFunctions.ReLUDeriv
            )
        )
        self.outputSize = size
        self.blankData()

    def refreshValues(self):
        self.values = list()
        for layer in self.weightLayers:
            self.values.append(
                ValueLayer(
                    layer.getShape()[0],
                    ActivationFunctions.ReLU,
                    ActivationFunctions.ReLUDeriv,
                )
            )

    def load(self, filename):
        with open(filename, "rb") as handle:
            self.weightLayers = pickle.load(handle)
        self.refreshValues()

    def save(self, filename):
        with open(filename, "wb") as handle:
            pickle.dump(
                self.weightLayers, handle, protocol=pickle.HIGHEST_PROTOCOL
            )

    def predict(self, inputData):
        # Invalid input handling
        if inputData.size != self.inputSize:
            print(
                f"Invalid input data size, {inputData.size} != {self.inputSize}"
            )
            return

        self.values[0].values = inputData.dot(
            self.weightLayers[0].weights.T
        )  # Multiplying the input data
        for i in range(1, len(self.values)):
            self.values[i].values = self.values[i - 1].values.dot(
                self.weightLayers[i].weights.T
            )
        return self.values[-1].values

    # Same as predict, but applies activation method
    # Perhaps combine the two into a single method with a flag argument?
    def forwardPropagate(self, inputData):
        # Invalid input handling
        if inputData.size != self.inputSize:
            print(
                f"Invalid input data size, {inputData.size} != {self.inputSize}"
            )
            return

        # print(f"{inputData}.dot({self.weightLayers[0].weights.T}")
        self.values[0].values = inputData.dot(
            self.weightLayers[0].weights.T
        )  # Multiplying the input data
        self.values[0].applyMethod()
        for i in range(1, len(self.values)):
            self.values[i].values = self.values[i - 1].values.dot(
                self.weightLayers[i].weights.T
            )
            self.values[i].applyMethod()
        return self.values[-1].values

    def fit_new(self):
        for sample in self.dataset:
            output = self.forwardPropagate(sample.input)
            self.values[-1].delta = sample.output - output
            print(f"\nOutput delta = {self.values[-1].delta}")

            # Calculate the delta of hidden layers
            for i in range(len(self.values) - 2, -1, -1):
                # print(f"Handling values[{i}]")
                print(
                    f"v[{i}].delta = "
                    f"v[{i + 1}].delta.dot(w[{i + 1}]) * v[{i}].deriv "
                )
                self.values[i].delta = (
                    self.values[i + 1].delta.dot(
                        self.weightLayers[i + 1].weights
                    )
                    * self.values[i].getAfterDeriv()
                )

            # Backpropagate
            for i in range(len(self.weightLayers) - 1, -1, -1):
                # print(f"Handling weights[{i}]")
                if i == 0:
                    print(f"w[{i}] += alpha * input.T.dot(v[{i}].delta).T")

                    # print(f"Turning 0 {self.weightLayers[i].weights}")
                    self.weightLayers[i].weights = (
                        self.weightLayers[i].weights
                        + ALPHA * sample.input.T.dot(self.values[i].delta).T
                    )
                    # print(f"into {self.weightLayers[i].weights}")
                else:
                    print(f"w[{i}] += alpha * v[{i-1}].T.dot(v[{i}].delta).T")

                    # print(f"Turning {self.weightLayers[i].weights}")
                    self.weightLayers[i].weights = (
                        self.weightLayers[i].weights
                        + ALPHA
                        * self.values[i - 1]
                        .values.T.dot(self.values[i].delta)
                        .T
                    )
                    # print(f"into {self.weightLayers[i].weights}")

    def fit(self):
        for sample in self.dataset:
            output = self.forwardPropagate(sample.input)
            delta = output - sample.output
            print(f"delta = {delta}")

            # todo: Replace weight with a reference
            # for weight in self.values[::-1]:
            #     wDelta = delta @ weight
            #     weight = weight - 0.05 * wDelta
            #     delta = delta @ weight

            # todo: Have another look at the values used, something's not
            #  right. Weight[0] is discarded.
            for n in range(len(self.values) - 1, -1, -1):
                if n > 0:
                    print(f"Handling values[{n}]")
                    wDelta = delta @ self.weightLayers[n].weights
                    # todo: temporary solution, replace with deriv
                    # wDelta = self.weightLayers[n].activationMethod(wDelta)
                else:

                    wDelta = delta.T @ sample.input
                delta = delta @ self.weightLayers[n].weights
                self.weightLayers[n].weights = (
                    self.weightLayers[n].weights - 0.05 * wDelta
                )

    def updateLatestDataManual(self):
        for i in range(len(self.dataset[-1].input[0])):
            self.dataset[-1].input[0][i] = float(input("Enter input value: "))
        print(self.dataset[-1].input[0])

        for i in range(len(self.dataset[-1].output[0])):
            self.dataset[-1].output[0][i] = float(input("Enter output value: "))
        print(self.dataset[-1].output[0])

    def addSampleManual(self):
        network.dataset.append(
            Data(
                np.ones((1, network.inputSize)),
                np.ones((1, network.outputSize)),
            )
        )
        network.updateLatestDataManual()

    def addSampleRandom(self):
        network.dataset.append(
            Data(
                np.random.rand(1, network.inputSize),
                np.random.rand(1, network.outputSize),
            )
        )

    def displayDataset(self):
        for data in network.dataset:
            print(f"Input:{data.input[0]}")
            print(f"Output:{data.output[0]}\n")

    def addSampleColour(self, r: float, g: float, b: float, colour: int):
        network.dataset.append(
            Data(
                np.zeros((1, network.inputSize)),
                np.zeros((1, network.outputSize)),
            )
        )

        self.dataset[-1].input[0][0] = r
        self.dataset[-1].input[0][1] = g
        self.dataset[-1].input[0][2] = b

        print(f"colour = {colour}")
        self.dataset[-1].output[0][colour - 1] = 1

        print(
            f"Appending input{self.dataset[-1].input[0]}, output = {self.dataset[-1].output[0]}"
        )

    def loadColourFile(self, filename):
        with open(filename, "r") as handle:
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
                print(
                    f"WRONG!, {np.argmax(result[0])} != {np.argmax(sample.output[0])}"
                )
        return float(correct / total * 100)

    def activationMethodTest(self):
        self.values[0].applyMethod()


"""
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
        for i in range(len(self.values) - 1, 0, -1):
            print(f"i = {i}")
            if i == len(self.values) - 1:
                delta = output_delta
            else:
                delta = np.dot(delta.T, self.values[i])  # Calculating the delta of the lower layer neurons
            if i == 0:
                weighted_delta = np.outer(delta, input)                # Calculating the final layer's delta
            else:
                weighted_delta = np.outer(delta, self.values[i-1])  # Calculating the given layer's delta
            self.values[i] = self.values[i] - 0.02 * weighted_delta        # Adjusting the values
"""

# todo: File handling
# todo: More advanced user features
#   Replace np.ones data with custom values
# todo: Fix fit() for multi-layer nets


inputData = np.ones((1, int(input("Enter input data size: "))))
firstLayerSize = int(input("Enter first layer size: "))
network = NeuralNetwork(inputData.size, firstLayerSize)
while True:
    print(
        "0 - Add quick layer\n"
        "1 - Add custom layer\n"
        "2 - Fit\n"
        "3 - Display\n"
        "4 - Predict\n"
        "5 - Save\n"
        "6 - Load\n"
        "7 - Overwrite latest data\n"
        "8 - Append new data\n"
        "9 - Load colour file (REQUIRES 3/4 I/O FORMAT)\n"
        "10- Validate colours (REQUIRES 3/4 I/O FORMAT)\n"
    )
    operation = int(input("Choose operation: "))
    if operation == 0:
        network.addLayer(int(input("Enter layer size: ")))
    if operation == 1:
        network.addLayerRange(
            int(input("Enter layer size: ")),
            int(input("Enter min weight value:")),
            int(input("Enter max weight value: ")),
        )
    if operation == 2:
        count = int(input("How many times? "))
        for i in range(count):
            network.fit_new()
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
        network.loadColourFile(str(input("Enter file name: ")))
    if operation == 10:
        print(f"{network.validateColours()}%")
    if operation == 11:
        network.addSampleRandom()
        network.displayDataset()
