import numpy as np
import memory_utils as mu
import galis_network as gn
import matplotlib.pyplot as plt

class GALISSeqNet:
    """
    Uses galis for seq network
    """
    def __init__(self, N, k_d=0, k_theta=0, k_w=1./3, beta_1=1, beta_2=1./2):
        # parameters as in galis references
        self.gnn = gn.GALISNN(N, k_d, k_theta, k_w, beta_1, beta_2)
        # key tracking for write and passive mode (reset by sequence training)
        self.first_key = mu.random_patterns(N,1)
        self.last_key = self.first()
        self.current_key = self.first()
    def layer_size(self):
        return self.gnn.N
    def next(self, key):
        # run dynamics until attractor changes
        self.gnn.set_pattern(key)
        while True:
            self.gnn.activate()
            next_key = self.gnn.get_pattern()
            if (next_key != key).any(): break
        if (self.last_key == key).all(): self.last_key = next_key
        return next_key
    def first(self):
        return self.first_key
    def train_sequence(self, memory_size, verbose=0):
        sequence = mu.random_patterns(self.layer_size(),memory_size)
        self.first_key = sequence[:,[0]]
        self.last_key = sequence[:,[0]]
        self.current_key = sequence[:,[0]]
        # one-shot associations
        for m in range(memory_size):
            self.gnn.set_pattern(sequence[:,[m]])
            if m > 0: self.gnn.associate()
            self.gnn.advance_tick_mark()

class GALISMemoryNet:
    """
    Backprop for k->v but GALIS for k->k
    """
    def __init__(self, N_kv, f_kv, df_kv, af_kv, N_g, memory_size, k_d=0, k_theta=0, k_w=1./3, beta_1=1./4, beta_2=1./2):
        # N[k]: size of k^th layer (len L+1)
        # f[k]: activation function of k^th layer (len L)
        # df[k]: derivative of f[k] (len L)
        # af[k]: inverse of f[k] (len L)
        self.L_kv = len(N_kv)-1
        self.N_kv = N_kv
        self.f_kv = f_kv
        self.df_kv = df_kv
        self.af_kv = af_kv
        self.W_kv = mu.init_randn_W(N_kv)
        self.gnn = gn.GALISNN(N_g, k_d, k_theta, k_w, beta_1, beta_2)
        self.memory_size = memory_size
        # key tracking for write and passive mode (reset by sequence training)
        self.first_key = mu.random_patterns(N_kv[0],1)
        self.last_key = self.first()
        self.current_key = self.first()
        # for batch update in passive mode
        self.dW_kv = {k: np.zeros(self.W_kv[k].shape) for k in self.W_kv}
        # for visualization (not operation!)
        self.sequence_error_history = np.empty((0,))
        self.write_error_history = np.empty((0,))
        self.cheat_error_history = np.empty((0,0))
        self.kv_cheat = {}
        self.failed_write_count = 0
    def layer_size(self):
        return self.N_kv[0]
    def read(self, key):
        return np.sign(mu.forward_pass(key, self.W_kv, self.f_kv)[self.L_kv])
    def write(self, key, value, eta=0.01, num_epochs=10000, verbose=0, term=0.5, batch=True):
        # stops if each actual is within term of target
        cheat_keys = self.kv_cheat.keys()
        cheat_error_history = np.nan*np.ones((len(self.kv_cheat), num_epochs))
        write_error_history = np.nan*np.ones((num_epochs,))
        history_fun = lambda e: np.fabs(e).max()
        # history_fun = lambda e: (e**2).sum()
        # train k -> v with gradient descent, incorporating passive loop
        # e_fun = lambda x, target: x - target # squared loss
        e_fun = lambda x, target: (x - target)*(np.fabs(x - target) > term) # rectified squared loss
        current_key = key
        dW = {k: np.zeros(self.W_kv[k].shape) for k in self.W_kv}
        for epoch in range(num_epochs):
            if (current_key == key).all():
                if batch:
                    # batch W update
                    for k in self.W_kv:
                        self.W_kv[k] += dW[k]
                        dW[k] *= 0
                x = mu.forward_pass(key, self.W_kv, self.f_kv)
                e = e_fun(x[self.L_kv], value) #x[self.L_kv] - value
                # early termination:
                if (np.fabs(x[self.L_kv] - value) < term).all(): break
                write_error_history[epoch] = history_fun(x[self.L_kv] - value)
                E = 0.5*((x[self.L_kv] - value)**2).sum()
            else:
                x = mu.forward_pass(current_key, self.W_kv, self.f_kv)
                e = e_fun(x[self.L_kv], np.sign(x[self.L_kv])) # x[self.L_kv] - np.sign(x[self.L_kv])
            # Progress update:
            if epoch % int(num_epochs/10) == 0:
                if verbose > 0: print('%d: %f'%(epoch,E))
            # Weight update:
            y = mu.backward_pass(x, e, self.W_kv, self.df_kv)
            G = mu.error_gradient(x, y)
            for k in self.W_kv:
                if batch:
                    dW[k] += - eta * G[k] # batch
                else:
                    self.W_kv[k] += - eta * G[k] # stochastic
            # Track write/any catastrophic forgetting
            for ck in range(len(cheat_keys)):
                x_ck = mu.forward_pass(mu.unhash_pattern(cheat_keys[ck]), self.W_kv, self.f_kv)
                e_ck = x_ck[self.L_kv] - self.kv_cheat[cheat_keys[ck]]
                cheat_error_history[ck, epoch] = history_fun(e_ck)
            # update current key
            if (current_key == self.last_key).all():
                current_key = self.first()
            else:
                current_key = self.next(current_key)
        # check for successful write
        x = mu.forward_pass(key, self.W_kv, self.f_kv)
        if (np.sign(x[self.L_kv]) != value).any():
            self.failed_write_count += 1
        # update cheats
        self.write_error_history = write_error_history[:epoch]
        self.cheat_error_history = cheat_error_history[:,:epoch]
        self.kv_cheat[mu.hash_pattern(key)] = value
    def next(self, key):
        # run dynamics until attractor changes
        self.gnn.set_pattern(key)
        while True:
            self.gnn.activate()
            next_key = self.gnn.get_pattern()
            if (next_key != key).any(): break
        if (self.last_key == key).all(): self.last_key = next_key
        return next_key
    def first(self):
        return self.first_key
    def train_sequence(self, memory_size, verbose=0):
        sequence = mu.random_patterns(self.layer_size(),memory_size)
        self.first_key = sequence[:,[0]]
        self.last_key = sequence[:,[0]]
        self.current_key = sequence[:,[0]]
        # one-shot associations
        for m in range(memory_size):
            self.gnn.set_pattern(sequence[:,[m]])
            if m > 0: self.gnn.associate()
            self.gnn.advance_tick_mark()
    def passive_ticks(self, num_ticks, eta=0.01, batch = True, verbose=0):
        for t in range(num_ticks):
            x = mu.forward_pass(self.current_key, self.W_kv, self.f_kv)
            value = np.sign(x[self.L_kv])
            # Progress update:
            e = x[self.L_kv] - value
            # Weight update:
            y = mu.backward_pass(x, e, self.W_kv, self.df_kv)
            G = mu.error_gradient(x, y)
            for k in self.W_kv:
                if batch:
                    self.dW_kv[k] += - eta * G[k]
                    if (self.current_key == self.last_key).all():
                        self.W_kv[k] += self.dW_kv[k]
                        self.dW_kv[k] *= 0
                else:
                    self.W_kv[k] += - eta * G[k]
            # passive advance
            if verbose > 0:
                print('passive tick %d:  k->v %s (|>| %f)'%(t,mu.patterns_to_ints(np.concatenate((self.current_key, value),axis=1)), np.fabs(x[self.L_kv]).min()))
            if (self.current_key == self.last_key).all():
                self.current_key = self.first()
            else:
                self.current_key = self.next(self.current_key)

def make_tanh_gmn(layer_size, memory_size, kv_layers=3):
    N_kv = [layer_size]*kv_layers
    L_kv = len(N_kv)-1
    f_kv = {k: np.tanh for k in range(1,L_kv+1)}
    df_kv = {k: mu.tanh_df for k in range(1,L_kv+1)}
    af_kv = {k: np.arctanh for k in range(1,L_kv+1)}
    N_g = layer_size
    # retrain sequence until no clobbers
    for tries in range(20):
        gmn = GALISMemoryNet(N_kv, f_kv, df_kv, af_kv, N_g, memory_size)
        gmn.train_sequence(memory_size,verbose=1)
        # mem check
        k = gmn.first()
        keys = np.empty((gmn.layer_size(), memory_size))
        keys[:,[0]] = k
        clobbered = np.zeros(memory_size, dtype=bool)
        for m in range(1,memory_size):
            k = gmn.next(k)
            keys[:,[m]] = k
            clobbered[:m] |= (keys[:,:m]==keys[:,[m]]).all(axis=0)
        seq_mem_check = (not clobbered.any())
        if seq_mem_check: break
    print(clobbered)
    assert(seq_mem_check) # if not, in trouble
    return gmn

def gmnet_test():
    num_patterns = 70
    net = make_tanh_gmn(layer_size = 32, memory_size=num_patterns*1, kv_layers=3)
    values = mu.random_patterns(net.layer_size(),num_patterns)
    keys = np.empty((net.layer_size(),num_patterns))
    k = net.first()
    for m in range(values.shape[1]):
        print('writing pattern %d'%m)
        keys[:,[m]] = k
        net.write(k,values[:,[m]],verbose=1,term=0.5,eta=0.001,num_epochs=10000)
        # net.passive_ticks(10000)
        k = net.next(k)
    # memory accuracy
    net_values = net.read(keys)
    print('final accuracy:')
    print(np.fabs(values-net_values).max(axis=0))
    # show write histories
    print('target vs read')
    print(values[:,-1].T)
    print(net.read(keys[:,-1]).T)
    print((values==net.read(keys)).all())
    h = [None,None]
    for c in range(net.cheat_error_history.shape[0]):
        h[0] = plt.plot(net.cheat_error_history[c,:],'r.')[0]
    h[1] = plt.plot(net.write_error_history,'b.')[0]
    # plt.plot(net.cheat_error_history.sum(axis=0) + net.write_error_history,'g.')
    # print(net.write_error_history)
    plt.xlabel('Training iteration')
    plt.ylabel('L_inf error on previous and current memories')
    plt.title('learning curves during write(k,v)')
    plt.legend(h,['||read(k_prev)-sign(read(k_prev))||','||read(k)-v||'])
    plt.show()


if __name__=="__main__":
    # make_tanh_gmn(layer_size=32, memory_size=40, kv_layers=3)
    gmnet_test()
