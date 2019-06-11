"""
Used to generate Figure 8 in the 2019 NVM paper
"""
import numpy as np
import matplotlib.pyplot as pt
import pickle as pk

np.set_printoptions(precision=4, linewidth=200)

def run_trial(N, P, T, verbose = False):

    sample = np.random.choice(P,T)
    if verbose:
        print("sample:")
        print(np.arange(T))
        print(sample)
    
    X_ = np.random.choice([-1,1], (N,P))
    X = X_[:, sample]
    Y = np.random.choice([-1,1], (N,T))
    
    W = np.zeros((N,N))
    for t in range(T):
        W += (Y[:,[t]] - W.dot(X[:,[t]])) * X[:,[t]].T / N
    
    net_err = 0
    for t in range(T):
        m_t = np.flatnonzero(sample == sample[t])[-1]
        err = N - (np.sign(Y[:,[m_t]]) == np.sign(W.dot(X[:,[t]]))).sum()
        net_err += err
        if verbose:
            print("t=%d ~ %d: err = %d" % (t, m_t, err))
    
    if verbose: print("net err = %d" % net_err)

    return net_err

# Ns = [32, 64, 128, 256, 512, 1024]
# Ns = [256, 512, 1024]
Ns = [64, 256, 1024]
T_ratios = [.05, .1, .15, .2, .25, .3]
P_ratios = [.05, .08, .11, .14, .17, .2]
reps = 30
avg_net_errs = {}
std_net_errs = {}

# Set this to True to actually run the trials and save the results
if False:
    for N in Ns:
        print(N)
        for T_ratio in T_ratios:
            for P_ratio in P_ratios:

                P = int(P_ratio * N)
                T = int(T_ratio * N)

                net_errs = []
                for rep in range(reps):
                    net_errs.append(run_trial(N, P, T))
    
                avg_net_errs[N, T_ratio, P_ratio] = np.mean(net_errs)
                std_net_errs[N, T_ratio, P_ratio] = np.std(net_errs)
    
    with open("rh_cap.pkl","w") as rh_cap: pk.dump((avg_net_errs, std_net_errs), rh_cap)

with open("rh_cap.pkl","r") as rh_cap: (avg_net_errs, std_net_errs) = pk.load(rh_cap)

# Set this to True to load and plot the results
if True:
    pt.figure(figsize=(6,2.5))
    sp = 1
    # for N in Ns:
    for N in [64, 256, 1024]:
    
        pt.subplot(1,3, sp)
        pt.title("N = %d" % N)
    
        x, y, e = [], [], []
        for T_ratio in T_ratios:
            T = int(T_ratio * N)
            for P_ratio in P_ratios:
                x.append(np.sqrt(P_ratio * T_ratio))
                y.append(100.*avg_net_errs[N, T_ratio, P_ratio] / T)
                e.append(100.*std_net_errs[N, T_ratio, P_ratio] / T)
                # y.append(100.*avg_net_errs[N, T_ratio, P_ratio] / N / T)
                # e.append(100.*std_net_errs[N, T_ratio, P_ratio] / N / T)

        pt.plot([.138, .138], [np.min(y), np.max(y)], 'k:')
        # pt.plot([np.min(x),np.max(x)], [100./N, 100./N], 'k--')
        pt.scatter(x, y, marker='o', color='k', facecolor='none')
        # pt.errorbar(x, y, yerr=e, color='k', marker='o', markerfacecolor='none',linestyle='none')
        
        # if sp == 1: pt.legend([".138", "1 error", "% error"])
        if sp == 2: pt.xlabel("$\sqrt{TP/N^2}$")
        if sp == 1: pt.ylabel("Mean Hamming Distance")
        # if sp in [1, 4]: pt.ylabel("Average % errors per t")
        sp += 1
    
    pt.tight_layout()
    pt.show()
