import itertools as it
import numpy as np

def nearest_power_of_2(i):
    return 2**int(np.ceil(np.log2(i)))

def _expand_hadamard(N):
    """
    Expand to NxN Hadamard matrix using Sylvester construction.
    N must be a power of two.
    """

    # Persistent Hadamard matrix
    H = getattr(_expand_hadamard, "H", np.array([[1]]))

    # Check for power of 2
    if not np.log2(N) == int(np.log2(N)):
        raise(Exception("N=%d is not a power of 2"%N))

    # Sylvester construction out to N
    while H.shape[0] < N:
        H = np.concatenate((
                np.concatenate((H, H),axis=1),
                np.concatenate((H,-H),axis=1),
            ), axis=0)
    
    # Save for sequel
    _expand_hadamard.H = H
    return H

def random_hadamard(N, P):
    """
    Create randomized hadamard matrix of size NxP.
    N must be a power of 2.
    If P > N, only N columns are returned.
    """
    
    # Expand as necessary
    H = _expand_hadamard(N)[:N,:min(N,P)]
    
    # Randomly negate rows
    R = np.sign(np.random.randn(H.shape[0],1)) * H

    # Randomly interchange N pairs of rows
    for _ in range(N):
        m, n = np.random.randint(N), np.random.randint(N)
        R[n,:], R[m,:] = R[m,:].copy(), R[n,:].copy()
    
    # # Interchange every pair of rows with some probability (N^2 time)
    # for (m,n) in it.combinations(range(R.shape[0]),2):
    #     if np.random.randn() > 0:
    #         R[n,:], R[m,:] = R[m,:].copy(), R[n,:].copy()

    return R

def random_orthogonal_patterns(N, P):
    """
    Create an NxP matrix of roughly orthogonal patterns.
    N must be a power of 2.
    If P > N, then orthogonality is only preserved within successive groups of N columns.
    """
    R = random_hadamard(N, P)
    while R.shape[1] < P:
        R = np.concatenate(
            (R, random_hadamard(N, P - R.shape[1])), axis=1)
    return R

if __name__ == "__main__":

    # H = random_hadamard(4,2)
    H = random_orthogonal_patterns(4,4)
    print(H)
    print(H.T.dot(H))

    H = random_orthogonal_patterns(4,4)
    print(H)
    print(H.T.dot(H))

    # H = random_hadamard(8,3)
    H = random_orthogonal_patterns(8,3)
    print(H)
    print(H.T.dot(H))

    H = random_orthogonal_patterns(4,6)
    print(H)
    print(H.T.dot(H))
