import numpy as np

"""
Activation functions
"""
def slog_f(x): return (-1.)**(x < 0)*np.log(np.fabs(x)+1.)
def slog_df(x): return 1./(np.fabs(x)+1.)
def slog_af(y): return (-1.)**(y < 0)*(np.exp(np.fabs(y))-1.)
def tanh_df(x): return 1. - np.tanh(x)**2.

"""
Activity pattern manipulations
"""
def int_to_pattern(N, i):
    return (-1)**(1 & (i >> np.arange(N)[:,np.newaxis]))
def patterns_to_ints(P):
    N = P.shape[0]
    ints = (2**np.arange(N)).dot(P < 1).flatten()
    ints[np.isnan(ints)] = -1
    return ints
def hash_pattern(p):
    return tuple(p.flatten())
def unhash_pattern(p):
    return np.array((p,)).T
