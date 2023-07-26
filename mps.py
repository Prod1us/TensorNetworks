import numpy as np


class MixedCanonical:
    def __init__(self, j, left_part, right_part, ortho_center):
        self.left_part = left_part  # left-canonical matrices
        self.right_part = right_part  # right-canonical matrices
        self.ortho_center = ortho_center  # orthogonality center at site j
        self.j = j

        self.rank = len(left_part) + len(right_part) + 1

    def is_left_canonical(self):
        return self.j == self.rank - 1

    def is_right_canonical(self):
        return self.j == 0

    def norm(self):
        return np.tensordot(self.ortho_center, np.conj(self.ortho_center), axes=((0, 1, 2), (0, 1, 2)))


# function returns mixed canonical MPS at site j of provided state
def decompose(state: np.array, j: int, truncate=False, trunc_bond_d=0) -> MixedCanonical:
    tensor_rank = int(np.log2(state.size))

    if tensor_rank - j < 1:
        raise Exception("Orthogonality center index is out of bounds")

    # we extend shape of our tensor for easier calculations (we add two 1d legs, one on the left and one on the right)
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
        if truncate and d > trunc_bond_d:
            d = trunc_bond_d
            u = u[:, :d]
            vh = vh[:d, :]
            s = s[:d]

        return u, s, vh

    # calculation of the left canonical part of the MPS
    for i in range(j):
        shape = state.shape
        state = np.reshape(state, (int(np.prod(shape[0:2])), int(np.prod(shape[2:]))))

        u, s, vh = SVD(state)
        bond_d = s.size

        u = np.reshape(u, (shape[0], 2, bond_d))
        state = np.diag(s) @ vh
        state = np.reshape(state, np.insert(shape[2:], 0, bond_d))

        left_mps_list.append(u)

    # calculation of the right canonical part of the MPS
    for i in range(tensor_rank - j - 1):
        shape = state.shape
        state = np.reshape(state, (int(np.prod(shape[:-2])), int(np.prod(shape[-2:]))))

        u, s, vh = SVD(state)
        bond_d = s.size

        vh = np.reshape(vh, (bond_d, 2, shape[-1]))
        state = u @ np.diag(s)
        state = np.reshape(state, np.append(shape[:-2], bond_d))

        right_mps_list.insert(0, vh)

    MPS = MixedCanonical(j, left_mps_list, right_mps_list, state)

    return MPS


# function returns expectation value of provided operator at site j
def local_1site_expectation_value(state: np.array, j: int, operator: np.array) -> float:
    ortho_center = decompose(state, j).ortho_center

    contraction = np.tensordot(np.conj(ortho_center), ortho_center, axes=((0, 2), (0, 2)))
    contraction = np.tensordot(contraction, operator, axes=((0, 1), (0, 1)))
    return np.real(contraction)


# function returns expectation value of provided operator at site j and j+1
def local_2site_expectation_value(state: np.array, j: int, operator: np.array) -> float:
    MPS = decompose(state, j)

    if MPS.is_left_canonical():
        raise Exception("j is a last element of the MPS")

    ortho_center = MPS.ortho_center
    next_tensor = MPS.right_part[0]

    left_part = np.tensordot(ortho_center, np.conj(ortho_center), axes=((0,), (0,)))
    right_part = np.tensordot(next_tensor, np.conj(next_tensor), axes=((2,), (2,)))
    ring = np.tensordot(left_part, right_part, axes=((1, 3), (0, 2)))
    ring = ring.transpose((0, 2, 1, 3))

    return np.real(np.tensordot(operator, ring, axes=((0, 1, 2, 3), (0, 1, 2, 3))))
