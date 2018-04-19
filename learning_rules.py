import numpy as np
from activator import *

def linear_solve(w, b, X, Y, actx, acty):
    dwb = np.linalg.lstsq(
        np.concatenate((X.T, np.ones((X.shape[1],1))), axis=1), # ones for bias
        acty.g(Y).T, rcond=None)[0].T
    dw, db =  dwb[:,:-1], dwb[:,[-1]]
    return dw, db

def hebbian(w, b, X, Y, actx, acty):
    N = X.shape[0]
    alpha = 2./(actx.on - actx.off)
    beta = (alpha * actx.off + 1)
    one = np.ones(X.shape)
    dw = acty.g(Y).dot(alpha**2 * X.T - alpha * beta * one.T) / N
    db = acty.g(Y).dot(- alpha * beta * X.T + beta**2 * one.T).dot(one[:,:1]) / N
    return dw, db

def flash_mem(w, b, X, Y, actx, acty, learning_rule, verbose=False):
    
    dw, db = learning_rule(w, b, X, Y, actx, acty)
    w, b = w + dw, b + db

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
    act = logistic_activator(0.05, N)
    X = np.empty((N,K))
    for k in range(K):
        X[:,[k]] = act.make_pattern()
    
    W, b = np.zeros((N,N)), np.zeros((N,1))
    W, b = hebbian(W, b, X[:,:-1], X[:,1:], act, act,)
    
    Y = act.f(W.dot(X[:,:-1]) + b)
    print(act.e(X[:,1:], Y))
