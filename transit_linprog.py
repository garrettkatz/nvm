import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as mp

np.set_printoptions(linewidth=200)

N = 4
K = 2
T = 4*K

V_seq = np.sign(np.random.rand(N,K) - .5)
V_x = V_seq[:,:-1]
V_y = V_seq[:,1:]
P = V_x.shape[1]
print(V_x.T)
print(V_y.T)

# linprog
W = np.empty((N,N))
w_ii = 1.25
a = np.sqrt(1. - 1./w_ii)
z = np.arctanh(a) - w_ii*a
stinq = 0.001 # strict inequality
for i in range(1):
    # minimize c^T x subject to
    # A_ub x <= b_ub
    # A_eq x == b_eq
    # returns object with field 'x'
    c = np.random.randn(N)
    A_ub = np.empty((P,N))
    b_ub = np.empty((P,1))
    for p in range(P):
        # openings
        if np.sign(V_x[i,p]) != np.sign(V_y[i,p]):
            A_ub[p,:] = np.sign(V_x[i,p])*V_x[:,p]
            b_ub[p,0] = -np.fabs(z) - stinq
        else:
            A_ub[p,:] = -np.sign(V_x[i,p])*V_x[:,p]
            b_ub[p,0] = np.fabs(z) - stinq
    # # zero diagonal
    A_eq=np.zeros((1,N))
    A_eq[0,i] = 1.
    b_eq = 0.
    bounds = (-1,1) # defaults are non-negative
    result = so.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='simplex', callback=None, options=None)
    print(A_eq, b_eq)
    print(A_ub, b_ub)
    print(c.T)
    print(result.x, result.success, result.message)
    W[i,:] = result.x
    W[i,i] = w_ii

# print(W)
# # WX = sY
# # X.T W.T = sY.T
# W = np.linalg.lstsq(X.T, np.arctanh(Y).T, rcond=None)[0].T

# # # W = 1.1*np.eye(N)
# # #np.random.randn(N,N)

# V = np.empty((N,T))
# V[:,[0]] = V_seq[:,[0]]
# for t in range(1,T):
#     V[:,[t]] = np.tanh(W.dot(V[:,[t-1]]))

# def wsc(X):
#     return (X-X.min())/(X.max()-X.min())
# def vsc(V):
#     return .5*(V+1.)
# def traj_signs(V):
#     V_s = [np.sign(V[:,[0]])]
#     for j in range(1,V.shape[1]):
#         if (np.sign(V[:,[j]]) != np.sign(V_s[-1])).any():
#             V_s.append(np.sign(V[:,[j]]))
#     return np.concatenate(V_s,axis=1)

# V_t = traj_signs(V)
# # print(V_t)
# # print(V_seq)
# # print(np.sign(V_seq) == np.sign(V_t))
# # print((np.sign(V_seq) == np.sign(V_t)).all())

# mp.subplot(1,4,1)
# mp.imshow(wsc(W),cmap="Greys")
# mp.subplot(1,4,2)
# mp.imshow(vsc(V),cmap="Greys")
# mp.subplot(1,4,3)
# mp.imshow(vsc(traj_signs(V)),cmap="Greys")
# mp.subplot(1,4,4)
# mp.imshow(vsc(V_seq),cmap="Greys")
# mp.show()
