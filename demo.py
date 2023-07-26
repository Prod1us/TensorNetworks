import mps
import numpy as np
import matplotlib.pyplot as plt


# function generates random spin one-half states of rank n
def random(n):
    x = np.random.rand(2 ** n) + 1.j * np.random.rand(2 ** n)
    x = x / np.sqrt(np.vdot(x, x))
    x = np.reshape(x, list(np.repeat(2, n)))
    return x


if __name__ == "__main__":
    pauliZ = np.array([[1., 0.], [0., -1.]])
    pauliZZ = np.tensordot(pauliZ, pauliZ, axes=0)

    # TODO check truncation

    psi = random(10)
    j = 4
    print("Expectation value of pauli Z operator in random state psi at j site: " +
          str(mps.local_1site_expectation_value(psi, j, pauliZ)))

    print("Expectation value of kronecker product of two pauli Z operators in random state psi at j and j + 1 site: " +
          str(mps.local_2site_expectation_value(psi, j, pauliZZ)))
