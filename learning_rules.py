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

def flash_mem(X, Y, activator, learning_rule, verbose=False):
    
    w, b = learning_rule(X, Y, activator)

    _Y = activator.f(w.dot(X) + b)
    diff_count = (np.ones(Y.shape) - activator.e(Y, _Y)).sum()

    if verbose:
        print("Flash residual max: %f"%np.fabs(Y - _Y).max())
        print("Flash residual mad: %f"%np.fabs(Y - _Y).mean())
        print("Flash diff count: %d"%(diff_count))

    return w, b, diff_count


if __name__ == "__main__":
    
    N = 8
    K = 3
    act = logistic_activator(0.05, (N,1))
    X = np.empty((N,K))
    for k in range(K):
        X[:,[k]] = act.make_pattern()
    
    W, b = logistic_hebbian(X[:,:-1], X[:,1:], act)
    
    Y = act.f(W.dot(X[:,:-1]) + b)
    print(act.e(X[:,1:], Y))
