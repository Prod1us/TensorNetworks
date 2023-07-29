import numpy as np

import decomposition
import mpo

from numpy.typing import NDArray
complex_arr_t = NDArray[np.complex128]


class MixedCanonical:
    def __init__(self, state: complex_arr_t, j: int, trunc_bond_d=None):
        self.j = j
        self.rank = int(np.log2(state.size))

        if self.rank - j < 1:
            raise Exception("Orthogonality center index is out of bounds")

        # we extend shape of our tensor for easier calc (we add two 1d legs,one on the left and one on the right)
        extension = np.ones(1)
        state = np.tensordot(extension, state, axes=0)
        state = np.tensordot(state, extension, axes=0)

        left_mps_list = []
        right_mps_list = []

        # svd decomposition with integrated bond dimension truncation
        def SVD(state0):
            u, s, vh = np.linalg.svd(state0, full_matrices=False)
            d = s.size

            # bond dimension truncation
            if trunc_bond_d is not None and d > trunc_bond_d:
                d = trunc_bond_d
                u = u[:, :d]
                vh = vh[:d, :]
                s = s[:d]

            return u, s, vh

        # calculation of the left canonical part of the MPS
        for _ in range(j):
            shape = state.shape
            state = np.reshape(state, (int(np.prod(shape[0:2])), int(np.prod(shape[2:]))))

            u, s, vh = SVD(state)
            bond_d = s.size

            u = np.reshape(u, (shape[0], 2, bond_d))
            state = np.diag(s) @ vh
            state = np.reshape(state, np.insert(shape[2:], 0, bond_d))

            left_mps_list.append(u)

        # calculation of the right canonical part of the MPS
        for _ in range(self.rank - j - 1):
            shape = state.shape
            state = np.reshape(state, (int(np.prod(shape[:-2])), int(np.prod(shape[-2:]))))

            u, s, vh = SVD(state)
            bond_d = s.size

            vh = np.reshape(vh, (bond_d, 2, shape[-1]))
            state = u @ np.diag(s)
            state = np.reshape(state, np.append(shape[:-2], bond_d))

            right_mps_list.insert(0, vh)

        self.left_part = left_mps_list
        self.right_part = right_mps_list
        self.ortho_center = state

    def is_left_canonical(self):
        return self.j == self.rank - 1

    def is_right_canonical(self):
        return self.j == 0

    def norm(self):
        return np.real(np.tensordot(self.ortho_center, np.conj(self.ortho_center), axes=((0, 1, 2), (0, 1, 2))))

    # method returns expectation value of provided operator at site j
    def ev_1site(self, operator: complex_arr_t) -> float:
        # from orthogonality center and its conjugate we create one 2-leg tensor
        contraction = np.tensordot(np.conj(self.ortho_center), self.ortho_center, axes=((0, 2), (0, 2)))
        # performing contraction of operator and this tensor
        contraction = np.tensordot(contraction, operator, axes=((0, 1), (0, 1)))
        return np.real(contraction)

    # method returns expectation value of provided operator at site j and j+1
    def ev_2site(self, operator: complex_arr_t) -> float:
        if self.is_left_canonical():
            raise Exception("j is a last element of the MPS")

        next_tensor = self.right_part[0]

        # from mps tensors and its conjugate we create "tensor ring"
        # with 4 legs around operator tensor and then perform contraction
        left_part = np.tensordot(self.ortho_center, np.conj(self.ortho_center), axes=((0,), (0,)))
        right_part = np.tensordot(next_tensor, np.conj(next_tensor), axes=((2,), (2,)))
        ring = np.tensordot(left_part, right_part, axes=((1, 3), (0, 2)))
        ring = ring.transpose((0, 2, 1, 3))

        return np.real(np.tensordot(operator, ring, axes=((0, 1, 2, 3), (0, 1, 2, 3))))

    # method returns expectation value ov the operator represented as MPO in our MPS state
    def ev_mpo(self, MPO: mpo.QTIM) -> float:
        L, R = decomposition.mps_mpo_contraction(self, MPO)
        return np.real(np.tensordot(L[0], R[-1], axes=((0, 1, 2), (0, 1, 2))))

    def __getitem__(self, i):
        if 0 > i >= self.rank:
            Exception("Index out of bounds")

        if i < self.j:
            return self.left_part[i]
        if i == self.j:
            return self.ortho_center
        if i > self.j:
            return self.right_part[i - self.j - 1]

    # with this method we transfer our orthogonality center to a new site using gauge transformations
    def change_j(self, j: int):
        if j == self.j:
            return

        old_j = self.j
        self.j = j

        if j > old_j:
            for _ in range(j - old_j):
                shape = self.ortho_center.shape
                ortho_center = np.reshape(self.ortho_center, (shape[0] * shape[1], shape[2]))
                q, r = np.linalg.qr(ortho_center)

                self.left_part.append(np.reshape(q, shape))
                self.ortho_center = np.tensordot(r, self.right_part.pop(0), axes=((1,), (0,)))
        else:
            for _ in range(old_j - j):
                shape = self.ortho_center.shape
                ortho_center = np.reshape(self.ortho_center, (shape[0], shape[1] * shape[2]))
                q, r = np.linalg.qr(ortho_center.transpose())

                self.right_part.insert(0, np.reshape(q.transpose(), shape))
                self.ortho_center = np.tensordot(self.left_part.pop(-1), r, axes=((2,), (1,)))
