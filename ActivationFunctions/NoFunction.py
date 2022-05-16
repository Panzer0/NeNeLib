class NoFunction:
    @staticmethod
    def function(x):
        return x

    @staticmethod
    def derivative(layer):
        return [[1 for _ in layer[0]]]
