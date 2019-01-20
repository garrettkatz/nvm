import numpy as np
import matplotlib.pyplot as pt
import pickle as pk

np.set_printoptions(precision=4, linewidth=200)

# g = np.arctanh
# rho = .99

g = lambda x: x
rho = 1

def run_trial(N, P, rho, verbose = False):

    X = np.random.choice([-1,1], (N,P)) * rho
    Y = np.random.choice([-1,1], (N,P)) * rho
    
    W = np.zeros((N,N))
    means = []
    T = P
    for t in range(T):
    
        W += (g(Y[:,[t]]) - W.dot(X[:,[t]])) * X[:,[t]].T / (N * rho**2)
        means.append( np.mean(g(Y[:,:t+1]) * W.dot(X[:,:t+1])) )

        if verbose: print("t=%d: ~%f" % (t, means[-1]))
    
    return means

if __name__ == "__main__":

    N, P = 50, 400
    means = run_trial(N, P, rho, verbose = True)

    pt.plot(means)
    pt.show()

