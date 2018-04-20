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

def dipole(w, b, X, Y, actx, acty):
    # only works for single x, y
    
    # map x, y to [-1,1]
    wx = 2/(actx.on - actx.off)
    bx = -(actx.on + actx.off)/(actx.on - actx.off)
    sx = np.sign(wx*X + bx)
    wy = 2/(acty.on - acty.off)
    by = -(acty.on + acty.off)/(acty.on - acty.off)
    sy = np.sign(wy*Y + by)
    
    # map result back to acty
    yw = (acty.g(acty.on) - acty.g(acty.off))/2
    yb = (acty.g(acty.on) + acty.g(acty.off))/2

    # final weights
    N = X.shape[0]
    one = np.ones((N,1))
    dw = yw*sy*sx.T*wx - w
    db = yw*sy*(sx.T.dot(one)*bx - (N-1)) + yb - b

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
    
    # # logistic hebbian
    # N = 8
    # K = 3
    # act = logistic_activator(0.05, N)
    # X = np.empty((N,K))
    # for k in range(K):
    #     X[:,[k]] = act.make_pattern()
    
    # W, b = np.zeros((N,N)), np.zeros((N,1))
    # W, b = hebbian(W, b, X[:,:-1], X[:,1:], act, act,)
    
    # Y = act.f(W.dot(X[:,:-1]) + b)
    # print(act.e(X[:,1:], Y))

    # dipole
    N = 4
    K = 1
    act = logistic_activator(0.05, N)
    # act = tanh_activator(0.05, N)
    X = np.empty((N,K))
    Y = X.copy()
    for k in range(K):
        X[:,[k]] = act.make_pattern()
        Y[:,[k]] = act.make_pattern()
    
    W, b = np.zeros((N,N)), np.zeros((N,1))
    W, b = dipole(W, b, X, Y, act, act)
    print("Y",Y.T)
    print("X",X.T)
    print("W,b")
    print(W)
    print(b)
    
    Y_ = act.f(W.dot(X) + b)
    print(act.e(Y_, Y))

    idx = (np.random.rand(N,1) > .5)
    print("idx",idx.T)
    print("X",X.T)
    for i in range(X.shape[0]):
        if idx[i]:
            if X[i,0] == act.on: X[i,0] = act.off
            else: X[i,0] = act.on
    print("X",X.T)
    Y_ = act.f(W.dot(X) + b)
    print(act.e(Y_, Y))
