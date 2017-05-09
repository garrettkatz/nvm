import numpy as np

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
        # D[k] = diag(df(Wx)
        k = self.L
        e = (x[k] - d)
        y = {}
        for k in range(self.L,0,-1):
            y[k] = self.df[k-1]((self.W[k-1]+self.O[k-1]).dot(x[k-1])) * e
            e = (self.W[k-1]+self.O[k-1]).T.dot(y[k])
        return y
    def error_gradient(self, x, y):
        return [y[k+1] * x[k].T for k in range(self.L)]
    def gradient_descent(self, x_0, d, eta=0.01, num_epochs=10000, verbose=2):
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
    def memorize(self, x_0, d, eta=0.01, num_epochs=2000, verbose=1):
        x = self.forward_pass(x_0)
        for k in range(L-1): self.O[k][:,:] = 0.
        self.O[L-1]  = (self.af[L-1](d) - self.W[L-1].dot(x[L-1])) * x[L-1].T / (x[L-1].T.dot(x[L-1]))
        for epoch in range(num_epochs):
            # progress
            if epoch == 0 or epoch % int(num_epochs/10) == 0:
                x = self.forward_pass(x_0)
                y = self.backward_pass(x,d)
                G = self.error_gradient(x,y)
                E = 0.5*((x[self.L]-d)**2).sum()
                M = 0.5*sum([(self.O[k]**2).sum() for k in range(self.L)])
                # P = sum([(G[k]*self.O[k]).sum() for k in range(self.L)])
                if verbose > 0: print('%d: E=%f,M=%f'%(epoch,E,M))
                if verbose > 1:
                    print(x[self.L].T)
                    print(d.T)
            # predictor
            x = self.forward_pass(x_0)
            y = self.backward_pass(x,d)
            G = self.error_gradient(x,y)
            H = sum([(G[k]*self.O[k]).sum() for k in range(self.L)])/sum([(G[k]*G[k]).sum() for k in range(self.L)])
            for k in range(self.L):
                self.O[k] *= 1 - eta
                self.O[k] += eta * G[k] * H
            # corrector
            x = self.forward_pass(x_0)
            y = self.backward_pass(x, d)
            G = self.error_gradient(x, y)
            for k in range(self.L):
                self.O[k] += -G[k]

if __name__=='__main__':
    f = np.tanh
    df = lambda x: 1 - np.tanh(x)**2
    af = np.arctanh
    N = [2,2,1]
    L = len(N)-1
    kvn = KVNet(N, [f]*L, [df]*L, [af]*L)
    x_0 = np.random.randn(N[0],1)
    d = np.array([[.5]])
    # kvn.gradient_descent(x_0, d)
    kvn.memorize(x_0, d)
    
    
    
    # eta = 0.01
    # num_epochs = 1000
    # for epoch in range(num_epochs):
    #     x = kvn.forward_pass(x_0)
    #     if epoch % (num_epochs/10) == 0:
    #         print('epoch %d: d = %f, x[L] = %f, E = %f'%(epoch, d[0,0], x[L][0,0], 0.5*((x[L]-d)**2).sum()))
    #     y = kvn.backward_pass(x, d)
    #     for k in range(kvn.L):
    #         dOk = -y[k+1] * x[k].T
    #         kvn.O[k] += eta * dOk
            

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
