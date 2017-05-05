import numpy as np
import matplotlib.pyplot as plt

N = 64
W = 1.1*np.identity(N) + 0.9*np.random.randn(N,N)/N

def complete_trajectory(W, v0, max_steps=1000):
    """
    Follows trajectory until fixed point or pseudo-cycle.
    Fixed point check: use rnn-fxpts or just small tolerance.
    Pseudo-cycle check: check that v within small tolerance of line segment connecting two previous consecutive points.
    Return status, trajectory
    """

v = 2*np.random.rand(N,1) - 1
T = 2000
V = np.empty((N,T))
for t in range(T):
    V[:,[t]] = v
    v = np.tanh(W.dot(v))

plt.subplot(2,1,1)
plt.imshow(V,cmap='gray')
plt.subplot(2,1,2)
plt.imshow(np.sign(V),cmap='gray')
plt.show()
