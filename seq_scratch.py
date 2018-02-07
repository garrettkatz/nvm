import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so

np.set_printoptions(linewidth=200, formatter = {'float': lambda x: '% .3f'%x})

N = 32
K = 4
T = 20
pad = .2
seq_noise = .0
perturb_frac = 0.0
stinq = .01
num_trials = 1
successes = 0
# a = .85
a = (N-.5)/N
w_ii = 1./(1. - a**2)
z = np.arctanh(a) - w_ii*a

do_print = True
do_show = True
# do_print = False
# do_show = False

learn1 = lambda X, Y: np.linalg.lstsq(X.T, np.arctanh(Y).T, rcond=None)[0].T
learn2 = lambda X, Y: np.arctanh(Y).dot(X.T/N)
learn3 = lambda X, Y: 10*Y.dot(X.T/N)
def learn4(X, Y):
    # Make W close to I by leveraging null space of X
    N, M = X.shape
    U, s, Vh = np.linalg.svd(X)
    Z = (Vh.T / s).dot(U[:,:M].T)
    aY = np.arctanh(Y)
    A = np.linalg.lstsq(U[:,M:], (np.eye(N) - aY.dot(Z)).T, rcond=None)[0].T
    W = np.concatenate((aY, A),axis=1).dot(np.concatenate((Z,U[:,M:].T), axis=0))
    return W

def learn5(X, Y):
    # Make W diagonal greater than 1 with linear program
    # generic:
        # minimize c^T x subject to
        # A_ub x <= b_ub
        # A_eq x == b_eq
        # returns object with field 'x'
    # instance:
        # minimize -I[i,:] w[i,:]' to get large diagonal
        # constrain X' w[i,:]'  == atanh(Y[i,:])' to get target dynamics
        # bound problem by bounding all W elements
    N = X.shape[0]
    W = np.empty((N,N))
    I = np.eye(N)
    for i in range(N):
        c = -I[i,:]
        A_eq, b_eq = X.T, np.arctanh(Y[i,:]).T
        A_ub, b_ub = None, None
        bounds = 1.1*np.array([-1,1]) # defaults are non-negative
        result = so.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='interior-point', callback=None, options=None)
        W[i,:] = result.x
    return W

def learn6(X, Y):
    # Make W diagonal fixed greater than 1, preserve dynamics and direct hump openings with linear program
    # generic:
        # minimize c^T x subject to
        # A_ub x <= b_ub
        # A_eq x == b_eq
        # returns object with field 'x'
    # instance:
        # random c
        # constrain X' w[i,:]'  == atanh(Y[i,:])' to get target dynamics
        # constrain I[i,:] w[i,:]'  == w_ii to get fixed dynamics
        # constrain ub inequalities to get humps
        # bound problem by bounding all W elements
    N = X.shape[0]
    W = np.empty((N,N))
    I = np.eye(N)
    for i in range(N):
        c = np.random.randn(N)
        A_eq, b_eq = np.concatenate((X.T, I[[i],:])), np.concatenate((np.arctanh(Y[i,:]).T, np.array([w_ii])))
        A_ub, b_ub = np.empty(X.shape).T, np.empty((X.shape[1],1))
        for p in range(X.shape[1]):
            if np.sign(X[i,p]) != np.sign(Y[i,p]):
                A_ub[p,:] = np.sign(X[i,p])*X[:,p]
                A_ub[p,i] = 0.
                b_ub[p,0] = -np.fabs(z) - stinq
            else:
                A_ub[p,:] = -np.sign(X[i,p])*X[:,p]
                A_ub[p,i] = 0.
                b_ub[p,0] = np.fabs(z) - stinq
        bounds = max(w_ii,2)*np.array([-1,1]) # defaults are non-negative
        result = so.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='interior-point', callback=None, options=None)
        W[i,:] = result.x
    return W

def learn7(X, Y, a=.5):
    # Make humpy W with linear program
    # generic linprog:
        # minimize c^T x subject to
        # A_ub x <= b_ub
        # A_eq x == b_eq
        # returns object with fields 'x', 'message'
    N = X.shape[0]
    W = np.empty((N,N))
    I = np.eye(N)
    w_ii = 1./(1. - a**2)
    bounds = w_ii*np.array([-1,1]) # default bounds are non-negative
    successes = 0
    for i in range(N):
        A_eq, b_eq = I[[i],:], np.array([0.]) # zero diagonal for \hat{w}
        delta = (X[i,:] != Y[i,:])
        A_ub = np.concatenate((
            np.sign(X[i:i+1,delta].T) * a*np.sign(X[:,delta].T), # openings
            -np.sign(X[i:i+1,~delta].T) * np.sign(X[:,~delta].T), # asymptotes
        ), axis=0)
        b_ub = np.concatenate((
            -np.ones(delta.sum())*np.fabs(z), # openings
            np.ones((~delta).sum())*np.fabs(z), # asymptotes
        ))
        # method = 'simplex'
        method = 'interior-point'
        # c = np.random.randn(N)
        # c = np.ones(N)
        c = -A_ub.mean(axis=0)
        result = so.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method, callback=None, options=None)
        # # repeat for equal |w_ij|?
        # c = np.sign(result.x)
        # result = so.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method, callback=None, options=None)
        W[i,:] = result.x
        W[i,i] = w_ii
        # print('%d: %s'%(i,result.message))
        successes += (result.status == 0)
    print("%d of %d successful lps"%(successes,N))
    return W

def learn8(X, Y, a=.9):
    # Make humpy W with linear program
    # Try to align hump directions with transits
    # generic linprog:
        # minimize c^T x subject to
        # A_ub x <= b_ub
        # A_eq x == b_eq
        # returns object with fields 'x', 'message'
    N = X.shape[0]
    W = np.empty((N,N))
    I = np.eye(N)
    w_ii = 1./(1. - a**2)
    bounds = w_ii*np.array([-1,1]) # default bounds are non-negative
    successes = 0
    c = -(Y-X).mean(axis=1)
    for i in range(N):
        A_eq, b_eq = I[[i],:], np.array([0.]) # zero diagonal for \hat{w}
        delta = (X[i,:] != Y[i,:])
        A_ub = np.concatenate((
            np.sign(X[i:i+1,delta].T) * a*np.sign(X[:,delta].T), # openings
            -np.sign(X[i:i+1,~delta].T) * np.sign(X[:,~delta].T), # asymptotes
        ), axis=0)
        b_ub = np.concatenate((
            -np.ones(delta.sum())*np.fabs(z), # openings
            np.ones((~delta).sum())*np.fabs(z), # asymptotes
        ))
        # method = 'simplex'
        method = 'interior-point'
        # c = np.random.randn(N)
        # c = np.ones(N)
        # c = -A_ub.mean(axis=0)
        result = so.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method, callback=None, options=None)
        # # repeat for equal |w_ij|?
        # c = np.sign(result.x)
        # result = so.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method, callback=None, options=None)
        W[i,:] = result.x
        W[i,i] = w_ii
        # print('%d: %s'%(i,result.message))
        successes += (result.status == 0)
    print("%d of %d successful lps"%(successes,N))
    return W

def learn9(X, Y, a=.9):
    # Make humpy W with linear program
    # Try to align hump directions with transits, using eq
    # generic linprog:
        # minimize c^T x subject to
        # A_ub x <= b_ub
        # A_eq x == b_eq
        # returns object with fields 'x', 'message'
    N = X.shape[0]
    W = np.empty((N,N))
    I = np.eye(N)
    w_ii = 1./(1. - a**2)
    bounds = w_ii*np.array([-1,1]) # default bounds are non-negative
    successes = 0
    c = -(Y-X).mean(axis=1)
    for i in range(N):
        delta = (X[i,:] != Y[i,:])
        A_eq = np.concatenate((
            np.sign(X[i:i+1,delta].T) * a*np.sign(X[:,delta].T), # openings
            I[[i],:], # zero-diagonal
        ), axis=0)
        b_eq = np.concatenate((
            -np.ones(delta.sum())*np.fabs(z), # openings
            np.zeros(1), # zero-diagonal
        ))
        A_ub = -np.sign(X[i:i+1,~delta].T) * np.sign(X[:,~delta].T) # asymptotes
        b_ub = np.ones((~delta).sum())*np.fabs(z) # asymptotes
        # method = 'simplex'
        method = 'interior-point'
        # c = np.random.randn(N)
        # c = np.ones(N)
        # c = -A_ub.mean(axis=0)
        result = so.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method, callback=None, options=None)
        # # repeat for equal |w_ij|?
        # c = np.sign(result.x)
        # result = so.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method, callback=None, options=None)
        W[i,:] = result.x
        W[i,i] = w_ii
        # print('%d: %s'%(i,result.message))
        successes += (result.status == 0)
    print("%d of %d successful lps"%(successes,N))
    return W

def learn_seqs(V, a=.9):
    # Make humpy W with linear program
    # Try to align hump directions with transits, using additional ub
    # needs to know delta index at both k and k-1, so train on whole sequences
    # V[s] is the s^th sequence
    # generic linprog:
        # minimize c^T x subject to
        # A_ub x <= b_ub
        # A_eq x == b_eq
        # returns object with fields 'x', 'message'
    N = V[0].shape[0]
    W = np.empty((N,N))
    I = np.eye(N)
    w_ii = 1./(1. - a**2)
    bounds = w_ii*np.array([-1,1]) # default bounds are non-negative
    successes = 0
    for i in range(N):
        A_ub, b_ub = [], []
        for s in range(len(V)):
            Vs = np.sign(V[s])
            for k in range(V[s].shape[1]-1):
                if Vs[i,k] != Vs[i,k+1]:
                    # asymptote intersection must be in i opening
                    A_ub.append(a * np.sign(Vs[i,k]) * Vs[:,[k]].T)
                    b_ub.append(np.fabs(z))
                    # but i hump must be close enough to ensure previous transit
                    if k > 0:
                        delta = np.flatnonzero(Vs[:,k-1] != Vs[:,k])
                        for j in delta:
                            v = Vs[:,[k]].copy()
                            v[j,0] -= N*(1-a)*Vs[j,k]
                            A_ub.append(-np.sign(Vs[i,k]) * v.T)
                            b_ub.append(np.fabs(z))                            
                else:
                    # corner must be blocked by i asymptote
                    A_ub.append(-np.sign(Vs[i,k]) * Vs[:,[k]].T)
                    b_ub.append(np.fabs(z))
        A_ub = np.concatenate(A_ub, axis=0)
        b_ub = np.array(b_ub)
        A_eq, b_eq = I[[i],:], np.zeros(1) # zero-diagonal
        # method = 'simplex'
        method = 'interior-point'
        # c = np.random.randn(N)
        # c = np.ones(N)
        c = -A_ub.mean(axis=0)
        result = so.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method, callback=None, options=None)
        # # repeat for equal |w_ij|?
        # c = np.sign(result.x)
        # result = so.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method, callback=None, options=None)
        W[i,:] = result.x
        W[i,i] = w_ii
        # print('%d: %s'%(i,result.message))
        successes += (result.status == 0)
    print("%d of %d successful lps"%(successes,N))
    return W

# learns = [learn1, learn4]
learns = [learn_seqs]

for trial in range(num_trials):

    V_seq = np.sign(np.random.randn(N,K)) * ((1-pad) + seq_noise*np.random.rand(N,K))
    X = V_seq.copy()
    Y = np.roll(V_seq, -1, axis=1)
    
    for learn in learns:
        if learn == learn_seqs:
            W = learn([V_seq])
        else:
            W = learn(X, Y)
        # Gating dynamical transitions by hump scaling
        G = 1*(np.ones((N,N)) - np.eye(N)) + 1*np.eye(N)
        W = W * G
        if do_print:
            print(W)
            print(np.linalg.matrix_rank(W))
            
    
    V = np.empty((N,T))
    V[:,[0]] = V_seq[:,[0]]*((-1.)**(np.random.rand(N,1) < perturb_frac)) * 1
    for t in range(1,T):
        V[:,[t]] = np.tanh(W.dot(V[:,[t-1]]))
        # V[:,[t]] = np.sign(W.dot(V[:,[t-1]]))
    
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
    
    def check_match(V_test, V_actual):
        match_start = -1
        match_end = -1
        T = V_actual.shape[1]
        K = V_test.shape[1]
        for j in range(T-K):
            if (V_actual[:,j:j+K] == V_test).all():
                match_start = j
                break
        if match_start > -1:
            idx = np.arange(T - match_start) % K
            mismatch = (V_actual[:,match_start:T] != V_test[:,idx]).any(axis=0)
            nz = np.flatnonzero(mismatch)
            if nz.size == 0: match_end = T
            else: match_end = nz[0] + match_start
        return match_start, match_end
    
    V_t = traj_signs(V)
    ms, me = check_match(np.sign(V_seq), V_t)
    print('match from %d to %d of %d'%(ms, me, T))
    if ms > -1: successes += 1
    
print('%d of %d successes'%(successes, num_trials))

# print(V_t)
# print(V_seq)
# print(np.sign(V_seq) == np.sign(V_t))
# print((np.sign(V_seq) == np.sign(V_t)).all())

if do_show:
    plt.subplot(1,5,1)
    plt.imshow(wsc(W),cmap="gray")
    plt.title('W')
    plt.subplot(1,5,2)
    plt.imshow(vsc(V),cmap="gray")
    plt.title('V')
    plt.subplot(1,5,3)
    plt.imshow(vsc(traj_signs(V)),cmap="gray")
    plt.title('sign(V)')
    plt.subplot(1,5,4)
    plt.imshow(vsc(V_seq),cmap="gray")
    plt.title('V seq')
    plt.subplot(1,5,5)
    plt.imshow(wsc(X.T.dot(X)/N),cmap="gray")
    plt.title('X.T.dot(X)/N')
    plt.show()
