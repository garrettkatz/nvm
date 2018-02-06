import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so

np.set_printoptions(linewidth=200, formatter = {'float': lambda x: '% .3f'%x})

N = 2
T = 50

I = np.eye(N)
W = np.array([[1, -1],[1, 1]])
a = .6
w_ii = 1/(1-a**2)
z = np.arctanh(a) - w_ii*a
h = 0.7
w_ij = np.fabs(z/(h*a + (1-h)*1))
G = w_ii*I + w_ij*(np.ones((N,N)) - I)
W = W * G
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
