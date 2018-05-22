import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as mp

np.set_printoptions(linewidth=200, precision=3)

N = 32
K = 4
D = 4
# D = 4/N

num_trials = 1
num_feas = 0
for trial in range(num_trials):

    # V_seq = np.sign(np.random.rand(N,K) - .5)
    V_seq = np.ones((N,K))
    for k in range(1,K):
        # i_delta = (np.random.rand(N) < D)
        i_delta = np.random.permutation(N)[:D]
        print(i_delta)
        V_seq[:,k] = V_seq[:,k-1]
        V_seq[i_delta,k] *= -1
        # V_seq[~i_delta,k] = V_seq[~i_delta,k-1]

    # # open
    # V_x = V_seq[:,:-1]
    # V_y = V_seq[:,1:]
    # P = V_x.shape[1]

    # closed
    V_x = V_seq
    V_y = np.roll(V_seq, -1, axis=1)
    P = V_x.shape[1]
    
    # linprog
    W = np.empty((N,N))
    w_ii = 5
    a = np.sqrt(1. - 1./w_ii)
    z = np.arctanh(a) - w_ii*a
    stinq = 0.001 # strict inequality
    successes, messages = [], []
    for i in range(N):
        # minimize c^T x subject to
        # A_ub x <= b_ub
        # A_eq x == b_eq
        # returns object with field 'x'
        c = np.random.randn(N)
        A_ub = np.empty((P,N))
        b_ub = np.empty((P,1))
        for p in range(P):
            # openings
            if np.sign(V_x[i,p]) != np.sign(V_y[i,p]):
                A_ub[p,:] = np.sign(V_x[i,p])*V_x[:,p]
                b_ub[p,0] = -np.fabs(z) - stinq
            else:
                A_ub[p,:] = -np.sign(V_x[i,p])*V_x[:,p]
                b_ub[p,0] = np.fabs(z) - stinq
        # # zero diagonal
        A_eq=np.zeros((1,N))
        A_eq[0,i] = 1.
        b_eq = 0.
        bounds = 2*np.array([-1,1]) # defaults are non-negative
        result = so.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='interior-point', callback=None, options=None)
        # print(A_eq, b_eq)
        # print(A_ub, b_ub)
        # print(c.T)
        # print(result.x, result.success, result.message)
        successes.append(result.success)
        messages.append(result.message)
        W[i,:] = result.x
        W[i,i] = w_ii
    
    # print(W)
    if not all(successes):
        for i in range(N):
            if not successes[i]: print("%d: %s"%(i, messages[i]))
    else:
        print("feasible transits")
        num_feas += 1

print("%d of %d feasible!"%(num_feas,num_trials))

print(((W - w_ii*np.eye(N)).min(),(W - w_ii*np.eye(N)).max()))

T = K*D
V = np.empty((N,T))
V[:,[0]] = V_seq[:,[0]]
for t in range(1,T):
    V[:,[t]] = np.tanh(W.dot(V[:,[t-1]]))

def wsc(X):
    # return (X-X.min())/(X.max()-X.min())
    return (X + np.fabs(X).max())/(2*np.fabs(X).max())
def vsc(V):
    return .5*(V+1.)
def traj_signs(V):
    V_s = [np.sign(V[:,[0]])]
    for j in range(1,V.shape[1]):
        if (np.sign(V[:,[j]]) != np.sign(V_s[-1])).any():
            V_s.append(np.sign(V[:,[j]]))
    return np.concatenate(V_s,axis=1)

V_ts = traj_signs(V)
# # print(V_ts)
print("Training:")
print(V_seq.T)
# # print(np.sign(V_seq) == np.sign(V_ts))
# # print((np.sign(V_seq) == np.sign(V_ts)).all())

mp.subplot(1,4,1)
mp.imshow(wsc(W),cmap="gray")
mp.title("W")
mp.subplot(1,4,2)
mp.imshow(vsc(V),cmap="gray")
mp.title("Actual")
mp.subplot(1,4,3)
mp.imshow(vsc(traj_signs(V)),cmap="gray")
mp.title("sign(Actual)")
mp.subplot(1,4,4)
mp.imshow(vsc(V_seq),cmap="gray")
mp.title("Training")
mp.show()
