import numpy as np
from activator import *

def linear_solve(X, Y, activator):
    W = np.linalg.lstsq(
        np.concatenate((X.T, np.ones((X.shape[1],1))), axis=1), # ones for bias
        activator.g(Y).T, rcond=None)[0].T
    return W[:,:-1], W[:,[-1]]

def tanh_hebbian(X, Y, activator):
    W = activator.g(Y).dot(X.T) / X.shape[0]
    b = np.zeros((Y.shape[0],1))
    return W, b

def logistic_hebbian(X, Y, activator):
    N = X.shape[0]
    W = 2*activator.g(Y).dot(2*X.T - np.ones(X.T.shape))/N
    b = -activator.g(Y).dot(2*X.T - np.ones(X.T.shape)).dot(np.ones((N,1)))/N
    return W, b

if __name__ == "__main__":
    
    N = 8
    K = 3
    act = logistic_activator(0.05, N)
    X = np.empty((N,K))
    for k in range(K):
        X[:,[k]] = act.make_pattern()
    
    W, b = logistic_hebbian(X[:,:-1], X[:,1:], act)
    
    Y = act.f(W.dot(X[:,:-1]) + b)
    print(act.e(X[:,1:], Y))
