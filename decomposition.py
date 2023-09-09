import numpy as np


# instead of contracting mpo and mps at one time we perform consequential contraction with cost ~ (bond d)^3
def mps_mpo_contraction(MPS, MPO):
    if MPS.rank != MPO.rank:
        Exception("uneven ranks of MPO and MPS")

    # Tensors resulting from contraction from the "left side"
    left_tensors = []

    # identity tensor needed to perform contraction for the most left tensors
    L = np.ones((1, 1, 1))

    for i in range(0, MPO.rank - 1):
        L = np.tensordot(L, np.conj(MPS[i]), axes=((0,), (0,)))
        L = np.transpose(L, (3, 2, 0, 1))
        L = np.tensordot(L, MPO[i], axes=((2, 1), (0, 3)))
        L = np.transpose(L, (0, 3, 2, 1))
        L = np.tensordot(L, MPS[i], axes=((3, 2), (0, 1)))
        left_tensors.append(L)

    # Tensors resulting from contraction from the "right side"
    right_tensors = []

    # identity tensor needed to perform contraction for the most right tensors
    R = np.ones((1, 1, 1))

    for i in reversed(range(1, MPO.rank)):
        R = np.tensordot(R, np.conj(MPS[i]), axes=((0,), (2,)))
        R = np.transpose(R, (2, 3, 0, 1))
        R = np.tensordot(R, MPO[i], axes=((1, 2), (3, 2)))
        R = np.transpose(R, (0, 2, 3, 1))
        R = np.tensordot(R, MPS[i], axes=((2, 3), (1, 2)))
        right_tensors.append(R)

    # lets adopt convention that R tensors are numerated from right to left as: 0, 1, ... n - 1

    return left_tensors, right_tensors
