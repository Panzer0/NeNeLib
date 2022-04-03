import numpy as np


def ReLU(x):
    return np.maximum(0, x)


def ReLUDeriv(x):
    return 1 if x > 0 else 0


def noMethod(x):
    return x


def noMethodDeriv(x):
    return 1
