import numpy as np


# function returns mixed canonical MPS at site j of provided state
def mixed_canonical(state: np.array, j: int, truncate=False, trunc_bond_d=0):
    tensor_rank = int(np.log2(state.size))

    if tensor_rank - j < 1:
        print("Error: Orthogonality center index is out of bounds")
        return

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

    return left_mps_list, [state], right_mps_list


