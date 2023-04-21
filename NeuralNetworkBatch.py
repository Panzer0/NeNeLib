import pickle
import numpy as np
import ActivationFunctions.ReLU
import ActivationFunctions.Sigmoid

from ActivationFunctions.SoftMax import SoftMax
from MNISTHandler import MNISTHandler
from NetworkStructure.DataBatch import DataBatch
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
        self.blank_data()

    # todo: On further inspection, this makes no sense.
    # todo: It doesn't seem to be used anywhere either.
    # TODO: Remove, I guess.
    def is_empty(self) -> bool:
        return len(self.weightLayers) > 0

    def has_data(self, target) -> bool:
        return len(target) > 0

    def blank_data(self):
        self.training = list()

    def get_output_layer(self):
        return self.values[-1]

    def display(self):
        for i, (w, v) in enumerate(zip(self.weightLayers, self.values)):
            print(f"{w} w[{i}]\n{v} v[{i}] ({v.activationFunction.__name__})")

    def add_layer(
        self,
        batchSize,
        size,
        minValue=WEIGHT_RANGE_LOWER,
        maxValue=WEIGHT_RANGE_UPPER,
    ):
        # Append a new weight layer with random values in the defined range
        weights = (maxValue - minValue) * np.random.rand(
            size, self.values[-1].getSize()
        ) + minValue
        self.weightLayers.append(WeightLayer(weights))

        # Set the former output layer's method to the default function
        self.values[-1].setMethod(DEFAULT_FUNCTION)

        # Append a new output value layer with no activation method
        self.values.append(ValueLayerBatch(batchSize, size, OUTPUT_FUNCTION))
        self.outputSize = size

        # Remove old data, which might no longer be suitable for the new shape
        self.blank_data()

    def refresh_values(self):
        # Generate empty value layers
        self.values = [
            ValueLayerBatch(BATCH_SIZE, layer.getShape()[0], DEFAULT_FUNCTION)
            for layer in self.weightLayers
        ]
        # Remove the final layer's activation method
        self.values[-1].setMethod()

    def load(self, filename):
        with open(filename, "rb") as handle:
            self.weightLayers = pickle.load(handle)
        self.refresh_values()
        self.training.clear()
        self.testing.clear()
        self.inputSize = self.weightLayers[0].getShape()[1]
        self.outputSize = self.values[-1].getSize()

    def save(self, filename):
        with open(filename, "wb") as handle:
            pickle.dump(
                self.weightLayers, handle, protocol=pickle.HIGHEST_PROTOCOL
            )

    def forward_propagate(self, inputData):
        if inputData.shape[1] != self.inputSize:
            print(
                f"Invalid input data size, {inputData.shape[1]} != {self.inputSize}"
            )
            return

        # Forward propagate input through the network
        # inputData is used to store the previous layer's values
        for i in range(len(self.values)):
            self.values[i].values = inputData.dot(
                self.weightLayers[i].weights.T
            )
            self.values[i].applyMethod()
            self.values[i].applyDropoutNewMask()
            inputData = self.values[i].values

        return self.values[-1].values

    def fit(self):
        for batch in self.training:
            output = self.forward_propagate(batch.input)
            self.values[-1].delta = (
                2 / self.outputSize * (output - batch.output)
            )
            if self.values[-1].activationFunction.__name__ == "SoftMax":
                self.values[-1].delta /= batch.output.shape[0]

            # Hidden layer delta calculation
            # todo: Make sure the range covers all values
            for i in reversed(range(len(self.values) - 1)):
                self.values[i].delta = (
                        self.values[i + 1].delta.dot(
                            self.weightLayers[i + 1].weights
                        )
                        * self.values[i].getAfterDeriv()
                )
                self.values[i].applyMaskToDelta()

            # Backpropagation
            for i in reversed(range(len(self.weightLayers))):
                grad = (
                    batch.input.T.dot(self.values[i].delta).T
                    if i == 0
                    else self.values[i - 1].values.T.dot(self.values[i].delta).T
                )
                self.weightLayers[i].weights -= ALPHA * grad

    def update_latest_data_manual(self, target):
        for i in range(len(target[-1].input[0])):
            target[-1].input[0][i] = float(input("Enter input value: "))
        print(target[-1].input[0])

        for i in range(len(target[-1].output[0])):
            target[-1].output[0][i] = float(input("Enter output value: "))
        print(target[-1].output[0])

    def add_sample_manual(self, target):
        target.append(
            DataBatch(
                np.ones((1, self.inputSize)),
                np.ones((1, self.outputSize)),
                BATCH_SIZE == 1,
            )
        )
        network.update_latest_data_manual(target)

    def add_sample_random(self, target):
        target.append(
            DataBatch(
                np.random.rand(1, self.inputSize),
                np.random.rand(1, self.outputSize),
                BATCH_SIZE == 1,
            )
        )

    def display_dataset(self, target):
        if target:
            print(target[0])
        else:
            print("Dataset is empty")

    def add_sample_colour(
        self, r: float, g: float, b: float, colour: int, target
    ):
        target.append(
            DataBatch(
                np.zeros((1, network.inputSize)),
                np.zeros((1, network.outputSize)),
                BATCH_SIZE == 1,
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

    def load_colour_file(self, filename, target):
        with open(filename, "r") as handle:
            data = list(map(float, handle.read().split()))

        for i in range(0, len(data), 4):
            r, g, b, out = data[i : i + 4]
            self.add_sample_colour(r, g, b, int(out), target)

    def validate_multi_class(self, target):
        total, correct = 0, 0
        for sampleBatch in target:
            resultBatch = self.forward_propagate(sampleBatch.input)
            for result, sample in zip(resultBatch, sampleBatch.output):
                total += 1
                correct += np.argmax(result) == np.argmax(sample)
        return float(correct / total * 100)

    def activation_method_test(self):
        self.values[0].applyMethod()

    def set_weights(self, index):
        self.weightLayers[index].weights = np.array(
            [
                [float(input("Enter weight value: ")) for _ in row]
                for row in self.weightLayers[index].weights
            ]
        )

    # Overwrites the train and test data with MNIST
    def load_MNIST(self):
        handler = MNISTHandler()

        # Load training data
        train_input = handler.get_train_input(TRAINING_SIZE)
        train_output = handler.get_train_output(TRAINING_SIZE)
        self.training = [
            DataBatch(
                train_input[i : i + BATCH_SIZE],
                train_output[i : i + BATCH_SIZE],
                BATCH_SIZE == 1,
            )
            for i in range(0, TRAINING_SIZE, BATCH_SIZE)
        ]

        # Load testing data
        test_input = handler.get_test_input(TEST_SIZE)
        test_output = handler.get_test_output(TEST_SIZE)
        self.testing = [
            DataBatch(
                test_input[i : i + BATCH_SIZE],
                test_output[i : i + BATCH_SIZE],
                BATCH_SIZE == 1,
            )
            for i in range(0, TEST_SIZE, BATCH_SIZE)
        ]

    def single_out_data(self, target):
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
            network.add_layer(BATCH_SIZE, int(input("Enter layer size: ")))
        elif operation == 1:
            network.add_layer(
                int(input("Enter layer size: ")),
                int(input("Enter min weight value: ")),
                int(input("Enter max weight value: ")),
            )
        elif operation == 2:
            if network.has_data(network.training):
                count = int(input("How many times? "))
                for i in range(count):
                    network.fit()
                    print(
                        f"{i}: {network.validate_multi_class(network.training)}%"
                    )
            else:
                print(NO_DATA_MESS)
        elif operation == 3:
            network.display_dataset(network.training)
            network.display()
        elif operation == 4:
            choice = int(input(TRAIN_OR_TEST_MESS))
            target = network.training if choice == 0 else network.testing

            if network.has_data(target):
                for sample in target:
                    print(network.forward_propagate(sample.input))
            else:
                print(NO_DATA_MESS)
        elif operation == 5:
            network.save("data.pickle")
        elif operation == 6:
            network.load("data.pickle")
        elif operation == 7:
            choice = int(input(TRAIN_OR_TEST_MESS))
            target = network.training if choice == 0 else network.testing
            if network.has_data(target):
                network.update_latest_data_manual(target)
                network.display_dataset(target)
            else:
                print(NO_DATA_MESS)
        elif operation == 8:
            choice = int(input(TRAIN_OR_TEST_MESS))
            target = network.training if choice == 0 else network.testing
            network.add_sample_manual(target)
            network.display_dataset(target)
        elif operation == 9:
            choice = int(input(TRAIN_OR_TEST_MESS))
            target = network.training if choice == 0 else network.testing
            network.add_sample_random(target)
            network.display_dataset(target)
        elif operation == 10:
            choice = int(input(TRAIN_OR_TEST_MESS))
            target = network.training if choice == 0 else network.testing
            network.load_colour_file(str(input("Enter file name: ")), target)
        elif operation == 11:
            choice = int(input(TRAIN_OR_TEST_MESS))
            target = network.training if choice == 0 else network.testing
            print(f"{network.validate_multi_class(target)}% correct")
        elif operation == 12:
            network.set_weights(int(input("Enter weight layer index: ")))
        elif operation == 13:
            network.load_MNIST()
        else:
            print("Invalid operation!")
