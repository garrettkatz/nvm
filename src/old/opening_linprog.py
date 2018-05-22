import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so

np.set_printoptions(linewidth=200, formatter = {'float': lambda x: '% .3f'%x})

N = 2
T = 50

I = np.eye(N)
a = .7
w_ii = 1/(1-a**2)
z = np.arctanh(a) - w_ii*a
# h = 0.5
# w_ij = np.fabs(z/(h*a + (1-h)*1))
# W = np.array([[1, -1],[1, 1]])
# G = w_ii*I + w_ij*(np.ones((N,N)) - I)
# W = W * G

# Make W diagonal greater than 1 with linear program
# generic:
    # minimize c^T x subject to
    # A_ub x <= b_ub
    # A_eq x == b_eq
    # returns object with field 'x'
# instance:
    # random c
    # bound problem by bounding all W elements
    # opening constraints:
    # i delta: sign(X[i,:]).T * X.T' * W[i,:].T  < -|z|
    # i square: -sign(X[i,:]).T * X.T' * W[i,:].T  < |z|
    # diagonal constraint:
    # I[i,:] * W[i,:].T == 0 (then overwrite wih w_ii)
    
X = np.array([[1,1],[-1,1],[-1,-1],[1,-1]]).T
Y = np.roll(X, -1, axis=1)
# X = X[:,:2]
# Y = Y[:,:2]
W = np.empty((N,N))
for i in range(N):
    A_eq, b_eq = None, None
    A_eq, b_eq = I[[i],:], np.array([0.])
    delta = (X[i,:] != Y[i,:])
    print(delta)
    A_ub = np.concatenate((
        np.sign(X[i:i+1,delta].T) * a*np.sign(X[:,delta].T),
        -np.sign(X[i:i+1,~delta].T) * np.sign(X[:,~delta].T),
    ), axis=0)
    b_ub = np.concatenate((
        -np.ones(delta.sum())*np.fabs(z),
        np.ones((~delta).sum())*np.fabs(z),
    ))
    bounds = 2*w_ii*np.array([-1,1]) # defaults are non-negative
    # method = 'simplex'
    method = 'interior-point'
    # c = np.random.randn(N)
    # c = np.ones(N)
    c = -A_ub.mean(axis=0)
    print(c)
    result = so.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method, callback=None, options=None)
    # # repeat for equal |w_ij|
    # c = np.sign(result.x)
    # result = so.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method, callback=None, options=None)
    W[i,:] = result.x
    W[i,i] = w_ii
    print('%d: %s'%(i,result.message))

A = np.sqrt(1 - 1/np.diagonal(W))

print(W)
print(A)
                
V = np.empty((N,T))
V[:,0] = np.array([1,1])
for t in range(1,T):
    V[:,[t]] = np.tanh(W.dot(V[:,[t-1]]))

v_i = np.linspace(-1,1,100)[1:-1]
for i in range(N):
    v_j = (np.arctanh(v_i) - W[i,i]*v_i)/W[i, 1-i]
    v_ij = np.empty((N,v_i.size))
    v_ij[i,:] = v_i
    v_ij[1-i,:] = v_j
    plt.plot(*v_ij, linestyle='--',color='gray')
    z_ = np.arctanh(A[i]) - W[i,i]*A[i]
    s_ = z_/(W[i,1-i]**2)
    w_ = np.concatenate((A[i]*I[[i],:].T, A[i]*I[[i],:].T+ (W*(1-I))[[i],:].T),axis=1)
    plt.plot(*w_, linestyle='-',color='gray')
    sw_ = np.concatenate((A[i]*I[[i],:].T, A[i]*I[[i],:].T+ s_*(W*(1-I))[[i],:].T),axis=1)
    plt.plot(*sw_, linestyle='-',color='k')
    print("z_%d = %f"%(i,z_))

plt.plot([0,0],[-1,1], linestyle='-',color='gray')
plt.plot([-1,1], [0,0], linestyle='-',color='gray')

plt.plot(*V, marker='o',linestyle='-',color='k')

plt.xlim([-1,1])
plt.ylim([-1,1])
plt.show()
