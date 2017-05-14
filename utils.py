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
def pattern_to_int(N, p):
    if np.isnan(p).any(): return -1
    else: return (2**np.arange(N)).dot(p < 1).flat[0]
def hash_pattern(p):
    return tuple(p.flatten())
def unhash_pattern(p):
    return np.array((p,)).T
