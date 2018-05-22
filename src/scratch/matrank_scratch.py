import numpy as np

N_G = 100
N_H = 20
N_GH = N_G + N_H
N_L3 = 60
TF = 10
rk = 2

A = np.random.randn(rk,TF) # coefficients expanding rank rk to size TF
X = np.concatenate((
    np.random.randn(N_GH+N_L3-1, rk).dot(A),
    np.ones((1,TF))
),axis=0)

_, s, vh = np.linalg.svd(X, full_matrices=False)
A = vh[s > s.max() * max(X.shape) * np.finfo(s.dtype).eps, :]

# pY = np.random.randn(N_GH, rk).dot(A)
pY = np.concatenate((
    -np.ones((N_G-1,TF)),
    np.ones((1,TF)),
    # np.zeros((N_G,TF)),
    np.random.randn(N_H, A.shape[0]).dot(A)
),axis=0)
M = np.linalg.lstsq(X.T, pY.T, rcond=None)[0].T

print('X rk, pY rk, resid')
print(np.linalg.matrix_rank(X))
print(np.linalg.matrix_rank(pY))
print(np.fabs(pY - M.dot(X)).max())

pY = M.dot(X)
M = np.linalg.lstsq(X.T, pY.T, rcond=None)[0].T
print('X rk, pY rk, resid')
print(np.linalg.matrix_rank(X))
print(np.linalg.matrix_rank(pY))
print(np.fabs(pY - M.dot(X)).max())

# W = np.linalg.lstsq(X.T, (np.arctanh(Y) - B).T, rcond=None)[0].T

