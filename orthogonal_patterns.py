import numpy as np

def make_random_orthogonal_patterns(N,P):
    """
    NxP matrix of P N-dimensional patterns.
    N must be a power of two and greater than or equal to P.
    """
    
    # Sylvester construction
    H = np.array([[1]])
    while H.shape[1] < P:
        H = np.concatenate((
                np.concatenate((H, H),axis=1),
                np.concatenate((H,-H),axis=1),
            ), axis=0)

    # Duplicate along left    
    H = H[:,:P]
    while H.shape[0] < N:
        H = np.concatenate((H,H), axis=0)
    
    if H.shape[0] != N: raise(Exception("N=%d is not a power of 2"%N))

    # Randomly negate and interchange rows
    H = np.sign(np.random.randn(N,1)) * H
    for n in range(N):
        m = np.random.randint(N)
        Hm, Hn = H[m,:].copy(), H[n,:].copy()
        H[n,:], H[m,:] = Hn, Hm
    
    return H

if __name__ == "__main__":

    H = make_random_orthogonal_patterns(8,3)
    print(H)
    print(H.T.dot(H))
