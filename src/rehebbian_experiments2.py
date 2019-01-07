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
    # sample[:P] = np.arange(P)
    
    X = np.random.choice([-1,1], (N,P))[:, sample] * rho
    Y = np.random.choice([-1,1], (N,T)) * rho
    
    W = np.zeros((N,N))
    means = []
    norms = []
    mt = np.arange(T)
    for t in range(T):
    
        W += (g(Y[:,[t]]) - W.dot(X[:,[t]])) * X[:,[t]].T / (N * rho**2)
        norms.append(np.fabs(W).max())

        # mt = t - (sample[:t+1,np.newaxis] == sample[np.newaxis,:t+1])[:,::-1].argmax(axis=1)
        # means.append( np.mean(g(Y[:,mt]) * W.dot(X[:,:t+1])) )
        # mt[:t][sample[:t] == sample[t]] = t # faster?
        # means.append( np.mean(g(Y[:,mt[:t+1]]) * W.dot(X[:,:t+1])) )

        # mt = np.empty(t+1, dtype=int)
        # for t1 in range(t+1):
        #     for t2 in range(t1, t+1):
        #         if (np.sign(X[:,t1]) == np.sign(X[:,t2])).all():
        #             mt[t1] = t2
        # means.append( np.mean(g(Y[:,mt]) * W.dot(X[:,:t+1])) )
        for t1 in range(t):
            if (np.sign(X[:,t1]) == np.sign(X[:,t])).all():
                mt[t1] = t
        means.append( np.mean(g(Y[:,mt[:t+1]]) * W.dot(X[:,:t+1])) )

        # print(t)
        # print(sample)
        # print(mt)
        # print(means[-1])
        # raw_input('.')

        if verbose: print("t=%d: ~%f" % (t, means[-1]))
    
    return means, norms

if __name__ == "__main__":

    pt.figure(figsize=(7,7))
    ax = [pt.subplot(2,1,sp) for sp in [1,2]]

    # linestyles = ['-', '--', '-.', ':']
    linestyles = ['-', '--', ':']
    # ratios = [.1, .25, .5, .9]
    ratios = [.14, .5, .9]
    # markers = list('osd^')
    markers = list('osd')
    leg = []

    num_trials = 30

    for i,pr in enumerate(ratios):

        ls = linestyles[i]

        # pt.subplot(3,1,i+1)
    
        # args = (N, P, T, errspace)
        args_list = [
            # (20, int(20 * pr), 401, 30),
            (50, int(50 * pr), 401, 30),
            (100, int(100 * pr), 401, 30),
            (200, int(200 * pr), 401, 30),
        ]
    
        for a, (N, P, T, errspace) in enumerate(args_list):
        
            m = markers[a]

            trial_means = []
            trial_norms = []
            for trial in range(num_trials):
                print("trial %d" % trial)
                run_means, run_norms = run_trial(N, P, T, rho, verbose = False)
                trial_means.append(run_means)
                trial_norms.append(run_norms)
        
            trial_means = np.array(trial_means)
            means = trial_means.mean(axis=0)
            stds = trial_means.std(axis=0)
    
            trial_norms = np.array(trial_norms)
            # norms = trial_norms.mean(axis=0)
            norms = trial_norms.max(axis=0)

            pt.sca(ax[0]) #pt.subplot(2,1,1)
            # print(means[-10:])
            # print(stds[-10:])
            # pt.errorbar(np.arange(0,T,errspace), means[::errspace], yerr = stds[::errspace], fmt=fmt)
            pt.plot(np.arange(0,T,errspace), means[::errspace], linestyle=ls, marker=m, color='k', markerfacecolor='w')

            pt.sca(ax[1]) #pt.subplot(2,1,2)
            pt.plot(np.arange(0,T,errspace), norms[::errspace], linestyle=ls, marker=m, color='k', markerfacecolor='w')
    
    
    pt.sca(ax[0]) #pt.subplot(2,1,1)
    # pt.ylabel("$E[\sigma^{-1}(y_i(m_T(t)))W_{i,:}(T)x(t)]$")
    pt.ylabel("Average product")

    pt.sca(ax[1]) #pt.subplot(2,1,2)
    pt.ylabel("max|W|")

    for sp in range(2):
    
        pt.sca(ax[sp]) #pt.subplot(2,1,sp)
        pt.xlabel("T")
        pt.xlim([0,T+100])
        # pt.ylim([2,7.5])
        pt.legend(
            [Line2D([0], [0], color='k', marker=m, markerfacecolor='w') for m in markers] + \
            [Line2D([0], [0], color='k', linestyle=ls) for ls in linestyles],
            # ["N=%d"%N for N in [20, 50, 100, 200]] +\
            ["N=%d"%N for N in [50, 100, 200]] +\
            ["P/N=%.2f"%pr for pr in ratios])

    # pt.legend(leg)
    pt.tight_layout()
    pt.show()


