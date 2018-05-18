import itertools as it
import numpy as np

def hadamard(N,P):
    """
    Make first P columns of NxN Hadamard matrix using Sylvester construction
    N must be a power of two and greater than or equal to P.
    """

    # Sylvester construction out to P columns
    H = np.array([[1]])
    while H.shape[1] < P:
        H = np.concatenate((
                np.concatenate((H, H),axis=1),
                np.concatenate((H,-H),axis=1),
            ), axis=0)

    # Continue to N rows without saving columns past P
    H = H[:,:P]
    while H.shape[0] < N:
        H = np.concatenate((H,H), axis=0)
    
    if H.shape[0] != N: raise(Exception("N=%d is not a power of 2"%N))
    
    return H

def randomize_hadamard(H):
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
