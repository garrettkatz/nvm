import numpy as np

N, P, rk = 100, 20, 15

X = np.random.randn(N,rk).dot(np.random.randn(rk,P))

U, S, Vh = np.linalg.svd(X, full_matrices=False)

Y = np.random.randn(N,rk).dot(Vh[:rk,:])

print(np.linalg.matrix_rank(X))
print(np.linalg.matrix_rank(Y))

W1 = np.linalg.lstsq(X.T, Y.T, rcond=None)[0].T
print(np.fabs(Y - W1.dot(X)).max())
print(np.linalg.norm(W1))

B = np.linalg.lstsq(Vh.T, Y.T, rcond=None)[0].T
print(np.fabs(Y - B.dot(Vh)).max())
print(np.fabs(B).max(axis=0))

B = np.linalg.lstsq(Vh[:rk,:].T, Y.T, rcond=None)[0].T
Si = np.diag(1./S[:rk])
W2 = B.dot(Si).dot(U[:,:rk].T)
print(np.fabs(Y - W2.dot(X)).max())
print(np.linalg.norm(W2))

print("Same thing!!")
