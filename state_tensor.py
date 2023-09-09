import numpy as np


# function generates random spin one-half states of rank n
def random(n):
    x = np.random.rand(2 ** n) + 1.j * np.random.rand(2 ** n)
    x = x / np.sqrt(np.vdot(x, x))
    x = np.reshape(x, list(np.repeat(2, n)))
    return x
