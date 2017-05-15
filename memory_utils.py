import numpy as np
import matplotlib.pyplot as plt

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

"""
Backprop helpers
Following Xie, Seung 2002
"""
def forward_pass(x_0, W, f):
    # x_0: input layer activity pattern
    # W[k]: weights from layer k-1 to k
    # f[k]: activation function at layer k
    # returns dict x[k]: activity at layer k
    x = {0: x_0}
    for k in range(1,len(W)+1):
        x[k] =  f[k]( W[k].dot(x[k-1]) ) 
    return x
def backward_pass(x, e, W, df):
    # x[k]: k^th layer activity pattern from forward pass
    # e: error vector at last layer, after differentiating loss function (e.g., x[L]-target for squared loss)
    # W[k]: weights from layer k-1 to k
    # df[k]: derivative of activation function at layer k
    # returns dict y[k]: 
    y = {}
    for k in range(len(x)-1, 0, -1):
        y[k] = df[k]( W[k].dot(x[k-1]) ) * e
        e = W[k].T.dot(y[k])
    return y
def error_gradient(x, y):
    # x,y from forward/backward pass
    # returns G[k]: error gradient wrt W[k]
    return {k: y[k] * x[k-1].T for k in range(1,len(x))}    
def init_randn_W(N):
    return {k: np.random.randn(N[k],N[k-1])/(N[k]*N[k-1]) for k in range(1,len(N))}
    
if __name__ == "__main__":
    
    N = [3]*3
    L = len(N)-1
    f = {k: np.tanh for k in range(1,L+1)}
    df = {k: tanh_df for k in range(1,L+1)}
    W = init_randn_W(N)
    # W = {k: np.eye(N[k],N[k-1]) for k in range(1,L+1)}
    print('W')
    for k in range(1,L+1):
        print(W[k])

    x_0 = np.array([[1,1,1]]).T
    z_L = 0.5*np.array([[-1,1,-1]]).T

    lcurve = []
    gcurve = []
    for epoch in range(2000):
        x = forward_pass(x_0, W, f)
        # print('x %d'%len(x))
        # for k in range(L+1):
        #     print(x[k].T)
            
        # print('z')
        # print(z_L.T)

        e = x[L] - z_L
        y = backward_pass(x, e, W, df)
        # print('y %d'%len(y))
        # for k in range(1,L+1):
        #     print(y[k].T)
    
        G = error_gradient(x, y)
        # print('G %d'%len(G))
        # for k in range(1,L+1):
        #     print(G[k])
    
        for k in range(1,L+1):
            W[k] += -0.001*G[k] #/np.sqrt((G[k]**2).sum())

        x_new = forward_pass(x_0, W, f)
        e_new = x_new[L] - z_L
    
        # print('error: old %f vs new %f'%(e.T.dot(e), e_new.T.dot(e_new)))
        # print(e.T)
        # print(e_new.T)
        
        lcurve.append((e**2).sum())
        gcurve.append(sum([(G[k]**2).sum() for k in range(1,L+1)]))

    print('W %d'%len(W))
    for k in range(1,L+1):
        print(W[k])
    plt.plot(np.arange(len(lcurve)),np.array(lcurve))
    plt.plot(np.arange(len(lcurve)),np.array(gcurve))
    plt.legend(['l','g'])
    plt.show()
    # raw_input('..')
    
