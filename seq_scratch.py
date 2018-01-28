import numpy as np
import matplotlib.pyplot as plt

N = 32
K = 4
T = 4*K

V_seq = np.sign(np.random.rand(N,K) - .5) * (.9 + .1*np.random.rand(N,K))
# X = V_seq.copy()
# Y = 1*(V_seq[:, np.arange(1,K+1) % K] - V_seq)
a = 0.05
X = (1-a)*V_seq[:,:-1] + a*V_seq[:, 1:]
Y = a*V_seq[:,:-1] + (1-a)*V_seq[:, 1:]

# WX = sY
# X.T W.T = sY.T
W = np.linalg.lstsq(X.T, np.arctanh(Y).T, rcond=None)[0].T

# # W = 1.1*np.eye(N)
# #np.random.randn(N,N)

V = np.empty((N,T))
V[:,[0]] = V_seq[:,[0]]
for t in range(1,T):
    V[:,[t]] = np.tanh(W.dot(V[:,[t-1]]))

def wsc(X):
    return (X-X.min())/(X.max()-X.min())
def vsc(V):
    return .5*(V+1.)
def traj_signs(V):
    V_s = [np.sign(V[:,[0]])]
    for j in range(1,V.shape[1]):
        if (np.sign(V[:,[j]]) != np.sign(V_s[-1])).any():
            V_s.append(np.sign(V[:,[j]]))
    return np.concatenate(V_s,axis=1)

V_t = traj_signs(V)
# print(V_t)
# print(V_seq)
# print(np.sign(V_seq) == np.sign(V_t))
# print((np.sign(V_seq) == np.sign(V_t)).all())

plt.subplot(1,4,1)
plt.imshow(wsc(W),cmap="Greys")
plt.subplot(1,4,2)
plt.imshow(vsc(V),cmap="Greys")
plt.subplot(1,4,3)
plt.imshow(vsc(traj_signs(V)),cmap="Greys")
plt.subplot(1,4,4)
plt.imshow(vsc(V_seq),cmap="Greys")
plt.show()
