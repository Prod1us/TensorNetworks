import numpy as np

import decomposition
import mps
import state_tensor

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigsh


# quantum transverse 1D Ising model Hamiltonian as a matrix product operator (MPO)
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

    def __setitem__(self, i, value):
        self.H[i] = value


# function calculates ground eigen system of the given matrix product operator (MPO)
def ground_system(MPO: QTIM) -> tuple[float, mps.MixedCanonical]:
    # init random mixed canonical matrix product state at site 0 (right canonical)
    psi = state_tensor.random(MPO.rank)
    MPS = mps.MixedCanonical(psi, 0, trunc_bond_d=2)

    # we decompose our "MPS-MPO-conj(MPS) sandwich" into consequentially contracted L and R tensors
    # they are gonna be our starting point (initial condition) in first left and right sweeps
    L, R = decomposition.mps_mpo_contraction(MPS, MPO)

    converged = False

    while not converged:
        left_sweep_energies = __sweep_from_left_to_right(L, R, MPO, MPS)
        right_sweep_energies = __sweep_from_right_to_left(L, R, MPO, MPS)

        # check convergence
        # I assume that state is converged if in the sweep from right to the left local energies are the same
        # (standard deviation is near numerical zero)
        converged = np.allclose(np.std(right_sweep_energies), 0)

    return right_sweep_energies[-2], MPS


# tensor contraction of effective hamiltonian h_n and A_n
def __A_hn_contraction(A_n, L, R, H_n):
    contraction = np.tensordot(L, A_n, axes=((2,), (0,)))
    contraction = np.tensordot(contraction, H_n, axes=((1, 2), (0, 1)))
    contraction = np.transpose(contraction, (0, 3, 2, 1))
    contraction = np.tensordot(contraction, R, axes=((2, 3), (1, 2)))

    return contraction


# function calculates ground state and energy of effective hamiltonian h_n at site n
# we are using Lanczos algorithm, so we only need to specify an application od our H_n operator on a vector
def __local_ground_system(L, R, H_n, A_n_shape):
    # purpose of this proxy function is to convert our tensor into vector shape back and forth
    # (eigsh() deals only with vectors)
    def apply_h_n(x):
        into_tensor = np.reshape(x, A_n_shape)

        contraction = __A_hn_contraction(into_tensor, L, R, H_n)
        back_to_vector = np.reshape(contraction, -1)
        return back_to_vector

    # size of the A_n as a vector
    dmx = np.prod(A_n_shape)

    A = LinearOperator(shape=(dmx, dmx), matvec=apply_h_n)
    energy, eigenstate = eigsh(A, 1, which='SA')

    return energy[0], np.reshape(eigenstate, A_n_shape)


def __sweep_from_left_to_right(L, R, MPO, MPS):
    rank = MPO.rank
    energy = []

    # initialize with identity tensor that is useful to perform boundary contractions
    L_tilde = np.ones((1, 1, 1))

    for i in range(0, rank - 1):
        # we obtain local ground state and ground energy of i's site of MPO
        local_energy, local_state = __local_ground_system(L_tilde, R[(rank - 2) - i], MPO[i], MPS[i].shape)
        energy.append(local_energy)

        # we perform QR decompo of obtained local ground state to make it unitary (R part goes to junk)
        local_state = np.reshape(local_state, (np.prod(local_state.shape[:-1]), local_state.shape[-1]))
        q, r = np.linalg.qr(local_state)

        # updating and overwriting L and MPS tensors
        new_A = np.reshape(q, MPS[i].shape)

        L_tilde = np.tensordot(L_tilde, np.conj(new_A), axes=((0,), (0,)))
        L_tilde = np.transpose(L_tilde, (3, 2, 0, 1))
        L_tilde = np.tensordot(L_tilde, MPO[i], axes=((2, 1), (0, 3)))
        L_tilde = np.transpose(L_tilde, (0, 3, 2, 1))
        L_tilde = np.tensordot(L_tilde, new_A, axes=((3, 2), (0, 1)))

        L[i] = L_tilde
        MPS[i] = new_A

    return energy


def __sweep_from_right_to_left(L, R, MPO, MPS):
    rank = MPO.rank
    energy = []

    # initialize with identity tensor that is useful to perform boundary contractions
    R_tilde = np.ones((1, 1, 1))

    for i in reversed(range(1, rank)):
        # we obtain local ground state and ground energy of i's site of  MPO
        local_energy, local_state = __local_ground_system(L[i - 1], R_tilde, MPO[i], MPS[i].shape)
        energy.append(local_energy)

        # we perform QR decompo (+ additional transposition) of obtained local ground state
        #  to make it unitary (R matrix goes to junk)
        local_state = np.reshape(local_state, (local_state.shape[0], np.prod(local_state.shape[-2:])))
        q, r = np.linalg.qr(np.transpose(local_state))
        q = np.transpose(q)

        # updating and overwriting R and MPS tensors
        new_A = np.reshape(q, MPS[i].shape)

        R_tilde = np.tensordot(R_tilde, np.conj(new_A), axes=((0,), (2,)))
        R_tilde = np.transpose(R_tilde, (2, 3, 0, 1))
        R_tilde = np.tensordot(R_tilde, MPO[i], axes=((1, 2), (3, 2)))
        R_tilde = np.transpose(R_tilde, (0, 2, 3, 1))
        R_tilde = np.tensordot(R_tilde, new_A, axes=((2, 3), (1, 2)))

        R[(rank - 1) - i] = R_tilde
        MPS[i] = new_A

    # additional optimization of the first tensor of the MPS (technically identical step as above)
    local_energy, local_state = __local_ground_system(np.ones((1, 1, 1)), R_tilde, MPO[0], MPS[0].shape)
    energy.append(local_energy)

    local_state = np.reshape(local_state, (local_state.shape[0], np.prod(local_state.shape[-2:])))
    q, r = np.linalg.qr(np.transpose(local_state))
    MPS[0] = np.reshape(np.transpose(q), MPS[0].shape)

    return energy
