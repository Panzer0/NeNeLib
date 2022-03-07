import numpy as np


class NeuralNetwork:
    def __init__(self, inputSize, firstLayerSize):
        self.values = list()
        self.weights = list()
        self.inputSize = inputSize

        self.weights.append(np.random.rand(firstLayerSize, inputSize))
        self.values.append(np.zeros(firstLayerSize))

    def getOutputLayer(self):
        return self.values[-1]

    def display(self):
        for weight, values, enumerate in zip(self.weights, self.values, range(len(self.weights))):
            print(f"{weight} w[{enumerate}] \n{values} v[{enumerate}]")

    def addLayer(self, size):
        print(f"Size = {self.values[-1].size}")
        self.weights.append(np.random.rand(size, self.values[-1].size))
        self.values.append(np.zeros((1, size)))

    def predict(self, inputData):
        # Invalid input handling
        if inputData.size != self.inputSize:
            print("Invalid input data size")
            return

        self.values[0] = inputData.dot(self.weights[0].T)  # Multiplying the input data
        for i in range(1, len(self.values)):
            self.values[i] = self.values[i - 1].dot(self.weights[i].T)
        return self.values[-1]

    def fit(self, input, expected):
        output = self.predict(input)
        delta = output - expected
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
                wDelta = delta.T @ input
            delta = delta @ self.weights[n]
            self.weights[n] = \
                self.weights[n] - 0.05 * wDelta

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
    print("1 - Add layer\n2 - Fit\n3 - Display\n4 - Predict")
    operation = int(input("Choose operation:"))
    if operation == 1:
        network.addLayer(int(input("Enter layer size")))
    if operation == 2:
        network.fit(inputData, np.ones((1, network.getOutputLayer().size)) * -1)
    if operation == 3:
        network.display()
    if operation == 4:
        print(network.predict(np.ones((1, network.inputSize))))

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
