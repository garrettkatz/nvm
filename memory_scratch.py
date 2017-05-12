import numpy as np
import matplotlib.pyplot as plt

class KVNet:
    """
    Feed-forward net for neural key-value memory
    """
    def __init__(self, N, f, df, af):
        # N[k]: size of k^th layer (len L+1)
        # f[k]: activation function of k^th layer (len L)
        # df[k]: derivative of f[k] (len L)
        # af[k]: inverse of f[k] (len L)
        self.L = len(N)-1 #
        self.N = N
        self.f = f
        self.df = df
        self.af = af
        # self.W = [np.random.randn(N[k+1],N[k])/(N[k+1]*N[k]) for k in range(self.L)]
        self.W = [np.eye(N[k+1],N[k]) for k in range(self.L)]
        self.O = [np.zeros((N[k+1],N[k])) for k in range(self.L)]
    def forward_pass(self, x_0):
        # x_0: input layer activity pattern
        x = {0: x_0}
        for k in range(self.L):
            x[k+1] = self.f[k]((self.W[k]+self.O[k]).dot(x[k]))
        return x
    def backward_pass(self, x, d):
        # x[k]: k^th layer activity pattern from forward pass
        # d: target pattern for L^th layer
        k = self.L
        e = (x[k] - d)
        y = {}
        for k in range(self.L,0,-1):
            y[k] = self.df[k-1]((self.W[k-1]+self.O[k-1]).dot(x[k-1])) * e
            e = (self.W[k-1]+self.O[k-1]).T.dot(y[k])
        return y
    def error_gradient(self, x, y):
        # x,y from forward/backward pass
        # returns error gradient wrt omega
        return [y[k+1] * x[k].T for k in range(self.L)]
    def gradient_descent(self, x_0, d, eta=0.01, num_epochs=10000, verbose=2):
        for k in range(L): self.O[k] *= 0.
        for epoch in range(num_epochs):
            x = self.forward_pass(x_0)
            y = self.backward_pass(x, d)
            if epoch % int(num_epochs/10) == 0:
                E = 0.5*((x[self.L]-d)**2).sum()
                if verbose > 0: print('%d: %f'%(epoch,E))
                if verbose > 1:
                    print(x[self.L].T)
                    print(d.T)
            G = self.error_gradient(x, y)
            for k in range(self.L):
                kvn.O[k] += - eta * G[k]
        # Persist weight changes
        for k in range(self.L):
            kvn.W[k] += kvn.O[k]
            kvn.O[k] *= 0
    def memorize(self, x_0, d, eta=0.01, num_epochs=5000, dot_term=0.999, verbose=1):
        x = self.forward_pass(x_0)
        for k in range(L-1): self.O[k] *= 0.
        self.O[L-1]  = (self.af[L-1](d) - self.W[L-1].dot(x[L-1])) * x[L-1].T / (x[L-1].T.dot(x[L-1]))
        learning_curve = []
        epoch = 0
        while True:
            # progress
            x = self.forward_pass(x_0)
            y = self.backward_pass(x,d)
            g = self.error_gradient(x,y)
            E = 0.5*((x[self.L]-d)**2).sum()
            O = sum([(self.O[k]**2).sum() for k in range(self.L)])
            G = sum([(g[k]**2).sum() for k in range(self.L)])
            GO = sum([(g[k]*self.O[k]).sum() for k in range(self.L)])
            if epoch % int(np.ceil(num_epochs/10. + 1)) == 0:
                # P = sum([(g[k]*self.O[k]).sum() for k in range(self.L)])
                if verbose > 0:
                    term = GO**2/(G*O) if G*O > 0 else 0
                    print('%d: E=%f,O=%f,term=%f'%(epoch,E,O,term))
                if verbose > 1:
                    print(x[self.L].T)
                    print(d.T)
            learning_curve.append((E,O,G,GO))
            # termination
            if epoch == num_epochs: break
            if G*O > 0 and GO**2/(G*O) > dot_term: break
            # predictor
            H = GO/G if G > 0 else 0
            for k in range(self.L):
                self.O[k] *= 1 - eta
                self.O[k] += eta * g[k] * H
            # corrector
            x = self.forward_pass(x_0)
            y = self.backward_pass(x, d)
            g = self.error_gradient(x, y)
            for k in range(self.L):
                self.O[k] += - eta * g[k]
            epoch += 1
        # Persist weight changes
        for k in range(self.L):
            kvn.W[k] += kvn.O[k]
            kvn.O[k] *= 0
        return learning_curve

# Activation functions
def slog_f(x): return (-1.)**(x < 0)*np.log(np.fabs(x)+1.)
def slog_df(x): return 1./(np.fabs(x)+1.)
def slog_af(y): return (-1.)**(y < 0)*(np.exp(np.fabs(y))-1.)
def tanh_df(x): return 1. - np.tanh(x)**2.

if __name__=='__main__':
    np.set_printoptions(linewidth=200)
    # Network size
    N = [10]*3
    L = len(N)-1
    M = 5 #sum(N) # num training examples
    keys = np.sign(np.random.randn(N[0],M))
    vals = 0.9*np.sign(np.random.randn(N[L],M)) #*(1./N[L])
    for num_epochs in [0, 10000]:
        for (f, df, af) in [
            # ([slog_f]*L, [slog_df]*L, [slog_af]*L),
            ([np.tanh]*L, [tanh_df]*L, [np.arctanh]*L)
        ]:
            kvn = KVNet(N, f, df, af)
            for m in range(M):            
                learning_curve = kvn.memorize(keys[:,[m]], vals[:,[m]],eta=0.001,num_epochs=num_epochs,verbose=1,dot_term=.99)
                E, O, G, GO = zip(*learning_curve)
                # # for k in range(L):
                # #     print(kvn.W[k])
                # # plt.ion()
                # plt.plot(E)
                # plt.plot(O)
                # plt.plot(G)
                # plt.legend(['Error','Omega norm','Gradient norm'])
                # plt.show()        
                outs = np.concatenate([kvn.forward_pass(keys[:,[_m]])[L] for _m in range(m+1)], axis=1)
                E = 0.5*((outs - vals[:,:m+1])**2).sum(axis=0)
                A = 1.0*(np.sign(outs) == np.sign(vals[:,:m+1])).sum(axis=0)/N[L]
                term = GO[-1]**2/(G[-1]*O[-1]) if G[-1]*O[-1] > 0 else 0
                print('%d: %d iters, O = %f, G = %f, term=%f, avg E = %f, avg A = %f'%(m,len(learning_curve), O[-1], G[-1], term, E.mean(),A.mean()))
                # print(E)
                print(A)
        # print(keys)
        # print(vals)
        # outs = np.concatenate([kvn.forward_pass(keys[:,[m]])[L] for m in range(M)], axis=1)
        # E = 0.5*((outs - vals)**2).sum(axis=0)
        # A = 1.0*(np.sign(outs) == np.sign(vals)).sum(axis=0)/N[L]
        # print(E)
        # print(A)
    
        # kvn.gradient_descent(x_0, d)
    
        # x_0, d = keys[:,[0]], vals[:,[0]]
        # learning_curve = kvn.memorize(x_0, d, eta=0.001,num_epochs=10000,verbose=1,dot_term=.99)
        # E, O, G, GO = zip(*learning_curve)
        # # plt.ion()
        # plt.plot(E)
        # plt.plot(O)
        # plt.plot(G)
        # plt.plot(np.array(GO)**2/(np.array(G)*np.array(O)))
        # plt.legend(['Error','Omega norm','Gradient norm','Term'])
        # plt.show()

def weight_update(W, G, X, Y):
    # learn X -> Y, each N x K
    # want
    #   np.tanh(W.dot(X)) = Y
    # or equivalently
    #   W.dot(X) = np.arctanh(Y)
    # competition of i and j at t measured by:
    #   C[i,j] = np.log(1+np.exp(-G[i,j])
    # where
    #   G[i,j] += (Y[i]*W[i,j]*X[j])
    # So, objective (obj), for each i separately:
    #   minimize ((dW[i,:]**2)*C[i,:]).sum()
    # subject to (sbt)
    #   (W+dW)[i,:].dot(X) - np.arctanh(Y[i,:]) = np.zeros((1,K))
    #   dW[i,:].dot(X) = np.arctanh(Y[i,:]) - W[i,:].dot(X)
    # Grads wrt to dW:
    #   obj: 2*(dW*C)[i,:]
    #   sbt: X.T
    # Lagrange says grad obj is lin comb of grad sbt:
    #   2*(dW*C)[i,:] = L.dot(X.T)
    # L is 1 x K lagrange multipliers.  Substitute out dW ito L in sbt:
    #   dW[i,:] = L.dot(X.T)/(2*C[i,:])
    #   (L.dot(X.T)/(2*C[i,:])).dot(X) = np.arctanh(Y[i,:]) - W[i,:].dot(X)
    #   L = mrdivide(np.arctanh(Y[i,:]) - W[i,:].dot(X), (X.T/(2*C[i,:])).dot(X))
    # Substitute back into obj:
    #   2*(dW*C)[i,:] = mrdivide(np.arctanh(Y[i,:]) - W[i,:].dot(X), (X.T/(2*C[i,:])).dot(X)).dot(X.T)
    #   dW[i,:] = mrdivide(np.arctanh(Y[i,:]) - W[i,:].dot(X), (X.T/(2*C[i,:])).dot(X)).dot(X.T)/(2*C[i,:])
    # If K == 1: (X.T/(2*C[i,:])).dot(X) is scalar, mrdivide is just scalar divide
    N = W.shape[0]
    T = X.shape[1]
    W = np.zeros((N,N,T))
    dW = np.zeros((N,N,T))
    G = np.zeros((N,N,T))
    for t in range(T):
        pass
