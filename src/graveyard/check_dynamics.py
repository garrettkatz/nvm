import scipy.linalg as spla
import numpy as np
import matplotlib.pyplot as plt

N = 3

# # random near I
# W = 1.1*np.identity(N) + 0.9*np.random.randn(N,N)/N

# expanding rotation with one preferred direction
S = 0.1*np.random.randn(N,N)/N # small random
S = S - S.T # make skew-symmetric
W = spla.expm(S) # make "rotation"

# plt.subplot(1,3,1)
# plt.imshow(S)
# plt.subplot(1,3,2)
# plt.imshow(W)

[w,vr] = spla.eig(W) # get eigenvectors
ix = ((vr-np.sign(vr))**2).sum(axis=0).argmax() # get one closest to vertex
# w = w/np.absolute(w) # normalize for rotation-like
w[ix] *= 1. + .000/N # expand on vertex aligned direction
W = vr.dot(np.diag(w)).dot(vr.T) # condition W

print(np.imag(W).max())
print(np.real(W).max())
W = np.real(W) # should be roughly real (some imaginary round-off)?

print(W)
# plt.subplot(1,3,3)
plt.imshow(W)

plt.show()

def complete_trajectory(W, v0, max_steps=1000):
    """
    Follows trajectory until fixed point or pseudo-cycle.
    Fixed point check: use rnn-fxpts or just small tolerance.
    Pseudo-cycle check: check that v within small tolerance of line segment connecting two previous consecutive points.
    Return status, trajectory
    """
    pass

# v = 2*np.random.rand(N,1) - 1
# T = 4000
# V = np.empty((N,T))
# for t in range(T):
#     V[:,[t]] = v
#     v = np.tanh(W.dot(v))

# V = np.repeat(V, 16, axis=0)
# plt.figure(figsize=(20,10))
# plt.subplot(2,1,1)
# plt.imshow(V,cmap='gray')
# plt.subplot(2,1,2)
# plt.imshow(np.sign(V),cmap='gray')
# plt.show()
