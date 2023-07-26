import mps
import numpy as np


def random(n):
  x = np.random.rand(2**n) + 1.j*np.random.rand(2**n)
  x = x/np.sqrt(np.vdot(x, x))
  x = np.reshape(x, list(np.repeat(2, n)))
  return x


def main():

if __name__ == "__main__":
    main()
