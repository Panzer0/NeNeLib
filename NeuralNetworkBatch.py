import pickle

import numpy as np

import ActivationFunctions.ReLU
import ActivationFunctions.Sigmoid
from ActivationFunctions.SoftMax import SoftMax
from ActivationFunctions.HyperbolicTangent import HyperbolicTangent
from ActivationFunctions.NoFunction import NoFunction
from MNISTHandler import MNISTHandler
from NetworkStructure.DataBatch import Data
from NetworkStructure.ValueLayerBatch import ValueLayerBatch
from NetworkStructure.WeightLayer import WeightLayer

ALPHA = 0.1
TRAINING_SIZE = 60_000
TEST_SIZE = 10_000

BATCH_SIZE = 100

DEFAULT_FUNCTION = ActivationFunctions.ReLU.ReLU
OUTPUT_FUNCTION = ActivationFunctions.SoftMax.SoftMax

WEIGHT_RANGE_LOWER = -0.1
WEIGHT_RANGE_UPPER = 0.1

# Pre-defined strings
TRAIN_OR_TEST_MESS = "Training (0) or testing (1) data? "
NO_DATA_MESS = "No data available "


class NeuralNetwork:
    def __init__(self, inputSize, firstLayerSize):
        self.values = list()
        self.weightLayers = list()
        self.training = list()
        self.testing = list()
        self.inputSize = inputSize
        self.outputSize = firstLayerSize

        self.weightLayers.append(
            WeightLayer(
                abs(WEIGHT_RANGE_UPPER - WEIGHT_RANGE_LOWER)
                * np.random.rand(firstLayerSize, inputSize)
                + WEIGHT_RANGE_LOWER
            )
        )
        self.values.append(
            ValueLayerBatch(BATCH_SIZE, firstLayerSize, OUTPUT_FUNCTION)
        )
        self.blankData()

    def isEmpty(self) -> bool:
        return len(self.weightLayers) > 0

    def hasData(self, target) -> bool:
        return len(target) > 0

    def blankData(self):
        self.training = list()

    def getOutputLayer(self):
        return self.values[-1]

    def display(self):
        for weight, values, index in zip(
            self.weightLayers, self.values, range(len(self.weightLayers))
        ):
            print(
                f"{weight} w[{index}]\n"
                f"{values} v[{index}] ({values.activationFunction.__name__})"
            )

    def addLayer(
        self,
        batchSize,
        size,
        minValue=WEIGHT_RANGE_LOWER,
        maxValue=WEIGHT_RANGE_UPPER,
    ):
        # Append a new weight layer with values in the defined range
        difference = abs(minValue - maxValue)
        self.weightLayers.append(
            WeightLayer(
                difference * np.random.rand(size, self.values[-1].getSize())
                + minValue
            )
        )
        # Set the former output layer's method to the default function
        self.values[-1].setMethod(DEFAULT_FUNCTION)
        # Append a new output value layer with no activation method
        self.values.append(ValueLayerBatch(batchSize, size, OUTPUT_FUNCTION))
        self.outputSize = size
        # Remove old data, which might no longer be suitable for the new shape
        self.blankData()

    def refreshValues(self):
        self.values = list()
        # Generate empty value layers
        for layer in self.weightLayers:
            self.values.append(
                ValueLayerBatch(
                    BATCH_SIZE, layer.getShape()[0], DEFAULT_FUNCTION
                )
            )
        # Remove the final layer's activation method
        self.values[-1].setMethod()

    def load(self, filename):
        with open(filename, "rb") as handle:
            self.weightLayers = pickle.load(handle)
        self.refreshValues()
        self.training.clear()
        self.testing.clear()
        self.inputSize = self.weightLayers[0].getShape()[1]
        self.outputSize = self.values[-1].getSize()

    def save(self, filename):
        with open(filename, "wb") as handle:
            pickle.dump(
                self.weightLayers, handle, protocol=pickle.HIGHEST_PROTOCOL
            )

    # Same as predict, but applies activation method
    # Perhaps combine the two into a single method with a flag argument?
    def forwardPropagate(self, inputData):
        # Invalid input handling
        if inputData.shape[1] != self.inputSize:
            print(
                f"Invalid input data size, {inputData.shape[1]} != {self.inputSize}"
            )
            return
        inputData = np.squeeze(inputData)
        self.values[0].values = inputData.dot(self.weightLayers[0].weights.T)
        # Multiplying the input data
        self.values[0].applyMethod()
        self.values[0].applyDropoutNewMask()
        for i in range(1, len(self.values)):
            self.values[i].values = self.values[i - 1].values.dot(
                self.weightLayers[i].weights.T
            )
            self.values[i].applyMethod()
            self.values[i].applyDropoutNewMask()
        return self.values[-1].values

    def fit(self):
        # todo: Add iteration over multiple batches
        for batch in self.training:
            output = self.forwardPropagate(batch.input)
            self.values[-1].delta = (
                2 / self.outputSize * (output - batch.output)
            )
            # print(f"delta = {self.values[-1].delta}")
            if self.values[-1].activationFunction.__name__ == "SoftMax":
                self.values[-1].delta /= batch.output.shape[0]

            # Calculate the delta of hidden layers
            for i in range(len(self.values) - 2, -1, -1):
                self.values[i].delta = self.values[i + 1].delta.dot(
                    self.weightLayers[i + 1].weights
                )
                self.values[i].delta = (
                    self.values[i].delta * self.values[i].getAfterDeriv()
                )
                self.values[i].applyMaskToDelta()

            # Backpropagate
            for i in range(len(self.weightLayers) - 1, -1, -1):
                if i == 0:
                    self.weightLayers[i].weights = (
                        self.weightLayers[i].weights
                        - ALPHA * batch.input.T.dot(self.values[i].delta).T
                    )
                else:
                    self.weightLayers[i].weights = (
                        self.weightLayers[i].weights
                        - ALPHA
                        * self.values[i - 1]
                        .values.T.dot(self.values[i].delta)
                        .T
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
            Data(np.ones((1, self.inputSize)), np.ones((1, self.outputSize)),)
        )
        network.updateLatestDataManual(target)

    def addSampleRandom(self, target):
        target.append(
            Data(
                np.random.rand(1, self.inputSize),
                np.random.rand(1, self.outputSize),
            )
        )

    def displayDataset(self, target):
        print(target[0])

    def addSampleColour(
        self, r: float, g: float, b: float, colour: int, target
    ):
        target.append(
            Data(
                np.zeros((1, network.inputSize)),
                np.zeros((1, network.outputSize)),
            )
        )

        print(f"Val = {target[-1].input[0][0]}")
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

    def validateMultiClass(self, target):
        total = 0
        correct = 0
        for sampleBatch in target:
            resultBatch = self.forwardPropagate(sampleBatch.input)
            for result, sample in zip(resultBatch, sampleBatch.output):
                total += 1
                if np.argmax(result) == np.argmax(sample):
                    # print("Correct!\n")
                    correct += 1
                # else:
                #     print(
                #         f"WRONG!, {np.argmax(result)} != {np.argmax(sample)}\n"
                #     )
        return float(correct / total * 100)

    def activationMethodTest(self):
        self.values[0].applyMethod()

    def setWeights(self, index):
        self.weightLayers[index].weights = np.array(
            [
                [float(input("Enter weight value: ")) for _ in row]
                for row in self.weightLayers[index].weights
            ]
        )

    # Overwrites the train and test data with MNIST
    def load_MNIST(self):
        # todo: Validate data size
        handler = MNISTHandler()

        self.training.clear()
        tempInput = handler.get_train_input(TRAINING_SIZE)
        tempOutput = handler.get_train_output(TRAINING_SIZE)
        self.training = np.squeeze(
            [
                Data(tempInput[low:high], tempOutput[low:high])
                for low, high in zip(
                    range(0, TRAINING_SIZE - BATCH_SIZE + 1, BATCH_SIZE),
                    range(BATCH_SIZE, TRAINING_SIZE + 1, BATCH_SIZE),
                )
            ]
        )

        self.testing.clear()
        tempInput = handler.get_train_input(TEST_SIZE)
        tempOutput = handler.get_train_output(TEST_SIZE)
        self.testing = np.squeeze(
            [
                Data(tempInput[low:high], tempOutput[low:high])
                for low, high in zip(
                    range(0, TEST_SIZE - BATCH_SIZE + 1, BATCH_SIZE),
                    range(BATCH_SIZE, TEST_SIZE + 1, BATCH_SIZE),
                )
            ]
        )

    def singleOutData(self, target):
        temp = target[0]
        target.clear()
        target.append(temp)


if __name__ == "__main__":
    inputData = np.ones((BATCH_SIZE, int(input("Enter input data size: "))))
    firstLayerSize = int(input("Enter first layer size: "))
    network = NeuralNetwork(inputData.shape[1], firstLayerSize)

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
            "11- Validate multi-class\n"
            "12- Set weights\n"
            "13- Load MNIST (REQUIRES 784/10 I/O FORMAT)\n"
        )
        operation = int(input("Choose operation: "))
        if operation == 0:
            network.addLayer(BATCH_SIZE, int(input("Enter layer size: ")))
        elif operation == 1:
            network.addLayer(
                int(input("Enter layer size: ")),
                int(input("Enter min weight value: ")),
                int(input("Enter max weight value: ")),
            )
        elif operation == 2:
            if network.hasData(network.training):
                count = int(input("How many times? "))
                for i in range(count):
                    network.fit()
                    print(
                        f"{i}: {network.validateMultiClass(network.training)}%"
                    )
            else:
                print(NO_DATA_MESS)
        elif operation == 3:
            network.displayDataset(network.training)
            network.display()
        elif operation == 4:
            choice = int(input(TRAIN_OR_TEST_MESS))
            target = network.training if choice == 0 else network.testing

            if network.hasData(target):
                for sample in target:
                    print(network.forwardPropagate(sample.input))
            else:
                print(NO_DATA_MESS)
        elif operation == 5:
            network.save("data.pickle")
        elif operation == 6:
            network.load("data.pickle")
        elif operation == 7:
            choice = int(input(TRAIN_OR_TEST_MESS))
            target = network.training if choice == 0 else network.testing

            if network.hasData(target):
                network.updateLatestDataManual(target)
                network.displayDataset(target)
            else:
                print(NO_DATA_MESS)
        elif operation == 8:
            choice = int(input(TRAIN_OR_TEST_MESS))
            target = network.training if choice == 0 else network.testing

            network.addSampleManual(target)
            network.displayDataset(target)
        elif operation == 9:
            choice = int(input(TRAIN_OR_TEST_MESS))
            target = network.training if choice == 0 else network.testing

            network.addSampleRandom(target)
            network.displayDataset(target)
        elif operation == 10:
            choice = int(input(TRAIN_OR_TEST_MESS))
            target = network.training if choice == 0 else network.testing

            network.loadColourFile(str(input("Enter file name: ")), target)
        elif operation == 11:
            choice = int(input(TRAIN_OR_TEST_MESS))
            target = network.training if choice == 0 else network.testing

            print(f"{network.validateMultiClass(target)}% correct")
        elif operation == 12:
            network.setWeights(int(input("Enter weight layer index: ")))
        elif operation == 13:
            network.load_MNIST()
        else:
            print("Invalid operation!")
