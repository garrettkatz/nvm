import numpy as np
import matplotlib.pyplot as pt
from matplotlib.lines import Line2D
import pickle as pk

np.set_printoptions(precision=4, linewidth=200)

g = np.arctanh
rho = .99

# g = lambda x: x
# rho = 1

def run_trial(N, P, T, rho, verbose = False):

    sample = np.random.choice(P,T)
    if verbose:
        print("sample:")
        print(np.arange(T))
        print(sample)
    sample[:P] = np.arange(P)
    
    X = np.random.choice([-1,1], (N,P))[:, sample] * rho
    Y = np.random.choice([-1,1], (N,T)) * rho
    
    W = np.zeros((N,N))
    means = []
    mt = np.arange(T)
    for t in range(T):
    
        W += (g(Y[:,[t]]) - W.dot(X[:,[t]])) * X[:,[t]].T / (N * rho**2)
        # mt = t - (sample[:t+1,np.newaxis] == sample[np.newaxis,:t+1])[:,::-1].argmax(axis=1)
        # means.append( np.mean(g(Y[:,mt]) * W.dot(X[:,:t+1])) )
        mt[:t][sample[:t] == sample[t]] = t # faster?
        means.append( np.mean(g(Y[:,mt[:t+1]]) * W.dot(X[:,:t+1])) )

        # print(t)
        # print(sample)
        # print(mt)
        # print(means[-1])
        # raw_input('.')

        if verbose: print("t=%d: ~%f" % (t, means[-1]))
    
    return means

if __name__ == "__main__":

    pt.figure(figsize=(7,4))

    linestyles = ['-', '--', '-.', ':']
    ratios = [.1, .25, .5, .9]
    markers = list('osd^')
    # markers = list('os')
    leg = []

    num_trials = 30

    for i,pr in enumerate(ratios):

        ls = linestyles[i]

        # pt.subplot(3,1,i+1)
    
        # args = (N, P, T, errspace)
        args_list = [
            (20, int(20 * pr), 301, 20),
            (50, int(50 * pr), 301, 20),
            (100, int(100 * pr), 301, 30),
            (200, int(200 * pr), 301, 30),
        ]
    
        for a, (N, P, T, errspace) in enumerate(args_list):
        
            m = markers[a]

            trial_means = []
            for trial in range(num_trials):
                print("trial %d" % trial)
                trial_means.append( run_trial(N, P, T, rho, verbose = False) )
        
            trial_means = np.array(trial_means)
            means = trial_means.mean(axis=0)
            stds = trial_means.std(axis=0)
    
            # print(means[-10:])
            # print(stds[-10:])
            # pt.errorbar(np.arange(0,T,errspace), means[::errspace], yerr = stds[::errspace], fmt=fmt)
            pt.plot(np.arange(0,T,errspace), means[::errspace], linestyle=ls, marker=m, color='k', markerfacecolor='w')
    
            # leg.append("N,P=%d,%d"%(N,P))
    
    pt.xlabel("T")
    # pt.ylabel("$E[\sigma^{-1}(y_i(m_T(t)))W_{i,:}(T)x(t)]$")
    pt.ylabel("Average product")
    
    pt.xlim([0,400])
    # pt.ylim([2,7.5])
    pt.legend(
        [Line2D([0], [0], color='k', marker=m, markerfacecolor='w') for m in markers] + \
        [Line2D([0], [0], color='k', linestyle=ls) for ls in linestyles],
        ["N=%d"%N for N in [20, 50, 100, 200]] +\
        ["P/N=%.2f"%pr for pr in ratios])

    # pt.legend(leg)
    pt.tight_layout()
    pt.show()


