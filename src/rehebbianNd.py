import numpy as np

np.set_printoptions(linewidth=200)

N = 7

X = (-1.) ** np.array([(np.arange(2**N) / 2**m) % 2 for m in range(N)])
print(X)

for k in range(N+1):
    idx = ((X == -1).sum(axis=0) == k)
    x = X[:,idx]
    print("")
    print("-------")
    print("")
    print(k)
    # print(x)
    print(x.dot(x.T))
    # print(x.T.dot(x))
    print(x.sum(axis=1)[:,np.newaxis])
