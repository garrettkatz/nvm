
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
