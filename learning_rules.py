import numpy as np
from activator import *

def linear_solve(X, Y, actx, acty):
    Wb = np.linalg.lstsq(
        np.concatenate((X.T, np.ones((X.shape[1],1))), axis=1), # ones for bias
        acty.g(Y).T, rcond=None)[0].T
    W, b =  Wb[:,:-1], Wb[:,[-1]]
    return W, b

def hebbian(X, Y, actx, acty):
    N = X.shape[0]
    alpha = 2./(actx.on - actx.off)
    beta = (alpha * actx.off + 1)
    one = np.ones(X.shape)
    W = acty.g(Y).dot(alpha**2 * X.T - alpha * beta * one.T) / N
    b = acty.g(Y).dot(- alpha * beta * X.T + beta**2 * one.T).dot(one[:,:1]) / N
    return W, b

def flash_mem(X, Y, actx, acty, learning_rule, verbose=False):
    
    w, b = learning_rule(X, Y, actx, acty)

    _Y = acty.f(w.dot(X) + b)
    diff_count = (np.ones(Y.shape) - acty.e(Y, _Y)).sum()

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
