import numpy as np
from activator import *

def linear_solve(X, Y, activator):
    W = np.linalg.lstsq(
        np.concatenate((X.T, np.ones((X.shape[1],1))), axis=1), # ones for bias
        activator.g(Y).T, rcond=None)[0].T
    return W[:,:-1], W[:,[-1]]

def tanh_hebbian(X, Y, activator):
    W = activator.g(Y).dot(
        np.concatenate((X.T, np.ones((X.shape[1],1))),axis=1) # ones for bias
        ) / X.shape[0]
    return W[:,:-1], W[:,[-1]]

def logistic_hebbian(X, Y, activator):
    N = X.shape[0]
    weights = 2*activator.g(Y).dot(2*X.T - np.ones(X.T.shape))/N
    bias = -activator.g(Y).dot(2*X.T - np.ones(X.T.shape)).dot(np.ones((N,1)))/N
    return weights, bias
