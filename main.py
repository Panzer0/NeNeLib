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
        self.training = list()
        self.testing = list()
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

    def hasData(self, target) -> bool:
        return len(target) > 0

    def blankData(self):
        self.training.clear()

    def getOutputLayer(self):
        return self.values[-1]

    def display(self):
        for weight, values, index in zip(
            self.weightLayers, self.values, range(len(self.weightLayers))
        ):
            print(f"{weight} w[{index}]\n" f"{values} v[{index}]")

    def addLayer(self, size, minValue=0, maxValue=1):
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
        self.training.clear()
        self.testing.clear()
        self.inputSize = self.weightLayers[0].getShape()[0]
        self.inputSize = self.values[-1].values.size
        print(f"Input = {self.inputSize}, output = {self.outputSize}")

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
        for sample in self.training:
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
                    print(f"w[{i}] += alpha * v[{i - 1}].T.dot(v[{i}].delta).T")

                    # print(f"Turning {self.weightLayers[i].weights}")
                    self.weightLayers[i].weights = (
                        self.weightLayers[i].weights
                        + ALPHA
                        * self.values[i - 1]
                        .values.T.dot(self.values[i].delta)
                        .T
                    )
                    # print(f"into {self.weightLayers[i].weights}")

    def fit_old(self):
        for sample in self.training:
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

    def updateLatestDataManual(self, target):
        for i in range(len(target[-1].input[0])):
            target[-1].input[0][i] = float(input("Enter input value: "))
        print(target[-1].input[0])

        for i in range(len(target[-1].output[0])):
            target[-1].output[0][i] = float(input("Enter output value: "))
        print(target[-1].output[0])

    def addSampleManual(self, target):
        target.append(
            Data(
                np.ones((1, self.inputSize)),
                np.ones((1, self.outputSize)),
            )
        )
        network.updateLatestDataManual(self.training)

    def addSampleRandom(self, target):
        target.append(
            Data(
                np.random.rand(1, self.inputSize),
                np.random.rand(1, self.outputSize),
            )
        )

    def displayDataset(self, target):
        for data in target:
            print(f"Input:{data.input[0]}")
            print(f"Output:{data.output[0]}\n")

    def addSampleColour(
        self, r: float, g: float, b: float, colour: int, target
    ):
        target.append(
            Data(
                np.zeros((1, network.inputSize)),
                np.zeros((1, network.outputSize)),
            )
        )

        target[-1].input[0][0] = r
        target[-1].input[0][1] = g
        target[-1].input[0][2] = b

        print(f"colour = {colour}")
        target[-1].output[0][colour - 1] = 1

        print(
            f"Appending input{target[-1].input[0]}, output = {target[-1].output[0]}"
        )

    def loadColourFile(self, filename, target):
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
                self.addSampleColour(r, g, b, int(out), target)
                flag = 0

    def validateColours(self, target):
        total = 0
        correct = 0
        for sample in target:
            total += 1
            result = self.forwardPropagate(sample.input)
            print(result)
            print(sample.output[0])
            print(f"Argmax = {np.argmax(result[0])}")

            if np.argmax(result[0]) == np.argmax(sample.output[0]):
                print("Correct!\n")
                correct += 1
            else:
                print(
                    f"WRONG!, {np.argmax(result[0])} != {np.argmax(sample.output[0])}\n"
                )
        return float(correct / total * 100)

    def activationMethodTest(self):
        self.values[0].applyMethod()

    def setWeights(self, index):
        self.weightLayers[index].weights = np.array(
            [
                [float(input("Enter weight value")) for weight in row]
                for row in self.weightLayers[index].weights
            ]
        )


# todo: File handling
# todo: Fix fit() for more advanced networks


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
        "9 - Append random data\n"
        "10- Load colour file (REQUIRES 3/4 I/O FORMAT)\n"
        "11- Validate colours (REQUIRES 3/4 I/O FORMAT)\n"
    )
    operation = int(input("Choose operation: "))
    if operation == 0:
        network.addLayer(int(input("Enter layer size: ")))
    if operation == 1:
        network.addLayer(
            int(input("Enter layer size: ")),
            int(input("Enter min weight value: ")),
            int(input("Enter max weight value: ")),
        )
    if operation == 2:
        if network.hasData(network.training):
            count = int(input("How many times? "))
            for i in range(count):
                network.fit_new()
                print("\n")
        else:
            print("No data available")
    if operation == 3:
        network.display()
    if operation == 4:
        choice = int(input("Training (0) or testing (1) data?"))
        target = network.training if choice == 0 else network.testing

        if network.hasData(target):
            for sample in target:
                print(network.forwardPropagate(sample.input))
        else:
            print("No data available")
    if operation == 5:
        network.save("data.pickle")
    if operation == 6:
        network.load("data.pickle")
    if operation == 7:
        choice = int(input("Training (0) or testing (1) data?"))
        target = network.training if choice == 0 else network.testing

        if network.hasData(target):
            network.updateLatestDataManual(target)
            network.displayDataset(target)
        else:
            print("No data available")
    if operation == 8:
        choice = int(input("Training (0) or testing (1) data?"))
        target = network.training if choice == 0 else network.testing

        network.addSampleManual(target)
        network.displayDataset(target)
    if operation == 9:
        choice = int(input("Training (0) or testing (1) data?"))
        target = network.training if choice == 0 else network.testing

        network.addSampleRandom(target)
        network.displayDataset(target)
    if operation == 10:
        choice = int(input("Training (0) or testing (1) data?"))
        target = network.training if choice == 0 else network.testing

        network.loadColourFile(str(input("Enter file name: ")), target)
    if operation == 11:
        choice = int(input("Training (0) or testing (1) data?"))
        target = network.training if choice == 0 else network.testing

        print(f"{network.validateColours(target)}%")
    if operation == 12:
        network.setWeights(int(input("Enter weight layer index")))
