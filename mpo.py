import numpy as np


# quantum transverse 1D Ising model hamiltonian as a matrix product operator (MPO)
import decomposition
import mps


class QTIM:
    pauliX = np.array([[0., 1.], [1., 0.]])
    pauliZ = np.array([[1., 0.], [0., -1.]])
    I = np.array([[1., 0.], [0., 1.]])

    def __init__(self, n: int, h: float, hz: float):
        self.rank = n
        self.h = h
        self.hz = hz

        H = []  # array of MPO tensors

        # first tensor of MPO
        H.append(np.zeros((1, 2, 3, 2)))
        H[0][0, :, 0, :] = - h * QTIM.pauliX - hz * QTIM.pauliZ
        H[0][0, :, 1, :] = -QTIM.pauliZ
        H[0][0, :, 2, :] = QTIM.I

        # middle part of MPO
        for j in range(n - 2):
            tensor = np.zeros((3, 2, 3, 2))
            tensor[0, :, 0, :] = QTIM.I
            tensor[1, :, 0, :] = QTIM.pauliZ
            tensor[2, :, 0, :] = -h * QTIM.pauliX - hz * QTIM.pauliZ
            tensor[2, :, 1, :] = - QTIM.pauliZ
            tensor[2, :, 2, :] = QTIM.I
            H.append(tensor)

        # last tensor of MPO
        H.append(np.zeros((3, 2, 1, 2)))
        H[-1][0, :, 0, :] = QTIM.I
        H[-1][1, :, 0, :] = QTIM.pauliZ
        H[-1][2, :, 0, :] = - h * QTIM.pauliX - hz * QTIM.pauliZ

        self.H = H

    # this method changes h parameter of our hamiltonian
    def change_h(self, h: float):
        new_value = - h * QTIM.pauliX - self.hz * QTIM.pauliZ

        self.H[0][0, :, 0, :] = new_value

        for j in range(self.rank - 2):
            self.H[1 + j][2, :, 0, :] = new_value

        self.H[-1][2, :, 0, :] = new_value

    # this method changes hz parameter of our hamiltonian
    def change_hz(self, hz: float):
        new_value = - self.h * QTIM.pauliX - hz * QTIM.pauliZ

        self.H[0][0, :, 0, :] = new_value

        for j in range(self.rank - 2):
            self.H[1 + j][2, :, 0, :] = new_value

        self.H[-1][2, :, 0, :] = new_value

    # this method changes h and hz parameters of our hamiltonian
    def change_h_hz(self, h: float, hz: float):
        new_value = - h * QTIM.pauliX - hz * QTIM.pauliZ

        self.H[0][0, :, 0, :] = new_value

        for j in range(self.rank - 2):
            self.H[1 + j][2, :, 0, :] = new_value

        self.H[-1][2, :, 0, :] = new_value

    def __getitem__(self, i):
        return self.H[i]
