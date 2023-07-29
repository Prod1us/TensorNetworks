import mps
import mpo
import numpy as np


# function generates random spin one-half states of rank n
def random(n):
    x = np.random.rand(2 ** n) + 1.j * np.random.rand(2 ** n)
    x = x / np.sqrt(np.vdot(x, x))
    x = np.reshape(x, list(np.repeat(2, n)))
    return x


if __name__ == "__main__":
    pauliZ = np.array([[1., 0.], [0., -1.]])
    pauliZZ = np.tensordot(pauliZ, pauliZ, axes=0)

    psi = random(8)
    j = 5
    MPS = mps.MixedCanonical(psi, j)
    MPS_trunc = mps.MixedCanonical(psi, j, 2)

    print("Expectation value of pauli Z operator in random state psi at j site: " +
          str(MPS.ev_1site(pauliZ)))

    print("Expectation value of pauli Z operator in random state psi at j site (with bond dimension truncation): " +
          str(MPS.ev_1site(pauliZ)))

    print("Expectation value of kronecker product of two pauli Z operators in random state psi at j and j + 1 sites: " +
          str(MPS.ev_2site(pauliZZ)))

    MPO = mpo.QTIM(8, 1, 0.1)

    print("Expectation value of QTIM Hamiltonian in random state psi: " +
          str(MPS.ev_mpo(MPO)))
