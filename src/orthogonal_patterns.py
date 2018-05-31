import itertools as it
import numpy as np

# Persistent Hadamard matrix
_H = np.array([[1]])

def expand_hadamard(N):
    """
    Expand to NxN Hadamard matrix using Sylvester construction.
    N must be a power of two.
    """

    # Check for power of 2
    if not np.log2(N) == int(np.log2(N)):
        raise(Exception("N=%d is not a power of 2"%N))

    # Sylvester construction out to N
    while _H.shape[0] < N:
        _H = np.concatenate((
                np.concatenate((_H, _H),axis=1),
                np.concatenate((_H,-_H),axis=1),
            ), axis=0)

def random_hadamard(N, P):
    """
    Create randomized hadamard matrix equivalent to H
    """
    
    # Randomly negate rows
    R = np.sign(np.random.randn(H.shape[0],1)) * H

    # Randomly interchange pairs of rows
    for (m,n) in it.combinations(range(R.shape[0]),2):
        if np.random.randn() > 0:
            R[n,:], R[m,:] = R[m,:].copy(), R[n,:].copy()
    
    return R

if __name__ == "__main__":

    H = randomize_hadamard(hadamard(8,3))
    print(H)
    print(H.T.dot(H))
