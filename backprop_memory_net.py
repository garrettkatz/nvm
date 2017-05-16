import numpy as np
import matplotlib.pyplot as plt
import memory_utils as mu

class BackPropKVNet:
    """
    Uses backprop for kv network, mock for sequence network
    """
    def __init__(self, N, f, df, af):
        # N[k]: size of k^th layer (len L+1)
        # f[k]: activation function of k^th layer (len L)
        # df[k]: derivative of f[k] (len L)
        # af[k]: inverse of f[k] (len L)
        self.L = len(N)-1
        self.N = N
        self.f = f
        self.df = df
        self.af = af
        self.W = mu.init_randn_W(N)
        # for passive ticks
        self.dW = {k: np.zeros(self.W[k].shape) for k in self.W}
        self.last_key = self.first()
        self.current_key = self.first()
        # for visualization (not operation!)
        self.write_error_history = np.empty((0,))
        self.cheat_error_history = np.empty((0,0))
        self.kv_cheat = {}
    def layer_size(self):
        return self.N[0]
    def read(self, key):
        return np.sign(mu.forward_pass(key, self.W, self.f)[self.L])
    def write(self, key, value, eta=0.01, num_epochs=100000, verbose=0, term=0.5):
        # self.write_vanilla(key, value, eta, num_epochs, verbose, term)
        self.write_passive(key, value, eta, num_epochs, verbose, term)
    def write_passive(self, key, value, eta=0.01, num_epochs=1000, verbose=0, term=0.5, batch=True):
        # stops if each actual is within term of target
        cheat_keys = self.kv_cheat.keys()
        cheat_error_history = np.nan*np.ones((len(self.kv_cheat), num_epochs))
        write_error_history = np.nan*np.ones((num_epochs,))
        history_fun = lambda e: np.fabs(e).max()
        # history_fun = lambda e: (e**2).sum()
        # train k -> v with gradient descent, incorporating passive loop
        current_key = key
        dW = {k: np.zeros(self.W[k].shape) for k in self.W}
        for epoch in range(num_epochs):
            if (current_key == key).all():
                if batch:
                    # batch W update
                    for k in self.W:
                        self.W[k] += dW[k]
                        dW[k] *= 0
                x = mu.forward_pass(key, self.W, self.f)
                e = x[self.L] - value
                # early termination:
                if (np.fabs(e) < term).all(): break
                write_error_history[epoch] = history_fun(e)
                E = 0.5*(e**2).sum()
            else:
                x = mu.forward_pass(current_key, self.W, self.f)
                e = x[self.L] - np.sign(x[self.L])
            # Progress update:
            if epoch % int(num_epochs/10) == 0:
                if verbose > 0: print('%d: %f'%(epoch,E))
            # Weight update:
            y = mu.backward_pass(x, e, self.W, self.df)
            G = mu.error_gradient(x, y)
            for k in self.W:
                if batch:
                    dW[k] += - eta * G[k] # batch
                else:
                    self.W[k] += - eta * G[k] # stochastic
            # Track write/any catastrophic forgetting
            for ck in range(len(cheat_keys)):
                x_ck = mu.forward_pass(mu.unhash_pattern(cheat_keys[ck]), self.W, self.f)
                e_ck = x_ck[self.L] - self.kv_cheat[cheat_keys[ck]]
                cheat_error_history[ck, epoch] = history_fun(e_ck)
            # update current key
            if (current_key == self.last_key).all():
                current_key = self.first()
            else:
                current_key = self.next(current_key)
        # update cheats
        self.write_error_history = write_error_history[:epoch]
        self.cheat_error_history = cheat_error_history[:,:epoch]
        self.kv_cheat[mu.hash_pattern(key)] = value
    def write_vanilla(self, key, value, eta=0.01, num_epochs=1000, verbose=0, term=0.5):
        cheat_keys = self.kv_cheat.keys()
        cheat_error_history = np.empty((len(self.kv_cheat), num_epochs))
        write_error_history = np.empty((num_epochs,))
        # history_fun = lambda e: np.fabs(e).max()
        history_fun = lambda e: (e**2).sum()
        # train k -> v with gradient descent, incorporating passive loop
        for epoch in range(num_epochs):
            x = mu.forward_pass(key, self.W, self.f)
            # early termination:
            if (np.fabs(x[self.L] - value) < term).all(): break
            # Progress update:
            e = x[self.L] - value
            if epoch % int(num_epochs/10) == 0:
                E = 0.5*(e**2).sum()
                if verbose > 0: print('%d: %f'%(epoch,E))
                if verbose > 1:
                    print(x[self.L].T)
                    print(value.T)
            # Weight update:
            y = mu.backward_pass(x, e, self.W, self.df)
            G = mu.error_gradient(x, y)
            for k in self.W:
                self.W[k] += - eta * G[k]
            # Track write/any catastrophic forgetting
            write_error_history[epoch] = history_fun(e)
            for ck in range(len(cheat_keys)):
                x_ck = mu.forward_pass(mu.unhash_pattern(cheat_keys[ck]), self.W, self.f)
                e_ck = x_ck[self.L] - self.kv_cheat[cheat_keys[ck]]
                cheat_error_history[ck, epoch] = history_fun(e_ck)
        # update cheats
        self.write_error_history = write_error_history[:epoch]
        self.cheat_error_history = cheat_error_history[:,:epoch]
        self.kv_cheat[mu.hash_pattern(key)] = value
    def next(self, key):
        next_key = mu.int_to_pattern(self.layer_size(), mu.patterns_to_ints(key)[0]+1)
        if (self.last_key == key).all(): self.last_key = next_key
        return next_key
    def first(self):
        return mu.int_to_pattern(self.layer_size(), 0)
    def passive_ticks(self, num_ticks, eta=0.01, batch = True, verbose=0):
        for t in range(num_ticks):
            # self.write(self.current_key, np.sign(self.read(self.current_key)))
            x = mu.forward_pass(self.current_key, self.W, self.f)
            value = np.sign(x[self.L])
            # Progress update:
            e = x[self.L] - value
            # Weight update:
            y = mu.backward_pass(x, e, self.W, self.df)
            G = mu.error_gradient(x, y)
            for k in self.W:
                if batch:
                    self.dW[k] += - eta * G[k]
                    if (self.current_key == self.last_key).all():
                        self.W[k] += self.dW[k]
                        self.dW[k] *= 0
                else:
                    self.W[k] += - eta * G[k]
            # passive advance
            if verbose > 0:
                print('passive tick %d:  k->v %s (|>| %f)'%(t,mu.patterns_to_ints(np.concatenate((self.current_key, value),axis=1)), np.fabs(x[self.L]).min()))
            if (self.current_key == self.last_key).all():
                self.current_key = self.first()
            else:
                self.current_key = self.next(self.current_key)

class BackPropSeqNet:
    """
    Uses backprop for seq network
    """
    def __init__(self, N, f, df, af):
        # N[k]: size of k^th layer (len L+1)
        # f[k]: activation function of k^th layer (len L)
        # df[k]: derivative of f[k] (len L)
        # af[k]: inverse of f[k] (len L)
        self.L = len(N)-1
        self.N = N
        self.f = f
        self.df = df
        self.af = af
        self.W = mu.init_randn_W(N)
        self.first_key = mu.random_patterns(N[0],1)
        self.write_error_history = np.empty((0,))
    def layer_size(self):
        return self.N[0]
    def next(self, key):
        return np.sign(mu.forward_pass(key, self.W, self.f)[self.L])
    def first(self):
        return self.first_key
    def pretrain(self, memory_size, eta=0.01, num_epochs=1000, term = 0.1, verbose=0):
        sequence = mu.random_patterns(self.layer_size(),memory_size)
        self.first_key = sequence[:,[0]]
        keys = sequence[:,:-1]
        next_keys = sequence[:,1:]
        error_history = np.empty((num_epochs,))
        # history_fun = lambda e: np.fabs(e).max()
        history_fun = lambda e: (e**2).sum()
        # train k -> v with gradient descent, incorporating passive loop
        for epoch in range(num_epochs):
            x = mu.forward_pass(keys, self.W, self.f)
            e = x[self.L] - next_keys
            E = 0.5*(e**2).sum()
            # early termination:
            if (np.fabs(e) < term).all(): break
            # Progress update:
            if epoch % int(num_epochs/10) == 0:
                if verbose > 0: print('%d: %f'%(epoch,E))
                if verbose > 1:
                    print(x[self.L].T)
                    print(value.T)
            # Weight update:
            y = mu.backward_pass(x, e, self.W, self.df)
            G = mu.error_gradient(x, y)
            for k in self.W:
                self.W[k] += - eta * G[k]
            # learning curve
            error_history[epoch] = history_fun(e)
        # update history
        self.error_history = error_history[:epoch]

class BackPropMemoryNet:
    """
    Uses backprop for both kv network and seq
    """
    def __init__(self, N_kv, f_kv, df_kv, af_kv, N_sq, f_sq, df_sq, af_sq, memory_size):
        # N[k]: size of k^th layer (len L+1)
        # f[k]: activation function of k^th layer (len L)
        # df[k]: derivative of f[k] (len L)
        # af[k]: inverse of f[k] (len L)
        self.L_kv = len(N_kv)-1
        self.N_kv = N_kv
        self.f_kv = f_kv
        self.df_kv = df_kv
        self.af_kv = af_kv
        self.L_sq = len(N_sq)-1
        self.N_sq = N_sq
        self.f_sq = f_sq
        self.df_sq = df_sq
        self.af_sq = af_sq
        self.memory_size = memory_size
        self.W_kv = mu.init_randn_W(N_kv)
        self.W_sq = mu.init_randn_W(N_sq)
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
        next_key = np.sign(mu.forward_pass(key, self.W_sq, self.f_sq)[self.L_sq])
        if (self.last_key == key).all(): self.last_key = next_key
        return next_key
    def first(self):
        return self.first_key
    def train_sequence(self, eta=0.01, num_epochs=1000, term = 0.1, verbose=0):
        sequence = mu.random_patterns(self.layer_size(),self.memory_size)
        keys = sequence[:,:-1]
        next_keys = sequence[:,1:]
        error_history = np.empty((num_epochs,))
        # history_fun = lambda e: np.fabs(e).max()
        history_fun = lambda e: (e**2).sum()
        # train k -> v with gradient descent, incorporating passive loop
        for epoch in range(num_epochs):
            x = mu.forward_pass(keys, self.W_sq, self.f_sq)
            e = x[self.L_sq] - next_keys
            E = 0.5*(e**2).sum()
            # early termination:
            if (np.fabs(e) < term).all(): break
            # Progress update:
            if epoch % int(num_epochs/10) == 0:
                if verbose > 0: print('%d: %f'%(epoch,E))
                if verbose > 1:
                    print(x[self.L_sq].T)
                    print(value.T)
            # Weight update:
            y = mu.backward_pass(x, e, self.W_sq, self.df_sq)
            G = mu.error_gradient(x, y)
            for k in self.W_sq:
                self.W_sq[k] += - eta * G[k]
            # learning curve
            error_history[epoch] = history_fun(e)
        # update history
        self.sequence_error_history = error_history[:epoch]
        # update key tracking
        self.first_key = sequence[:,[0]]
        self.last_key = sequence[:,[0]]
        self.current_key = sequence[:,[0]]
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

def make_tanh_kvn(layer_sizes):
    N = layer_sizes
    L = len(N)-1
    f, df, af = [np.tanh]*L, [mu.tanh_df]*L, [np.arctanh]*L
    f = {k: np.tanh for k in range(1,L+1)}
    df = {k: mu.tanh_df for k in range(1,L+1)}
    af = {k: np.arctanh for k in range(1,L+1)}
    kvn = BackPropKVNet(N=N,f=f,df=df,af=af)
    return kvn

def make_tanh_sqn(layer_sizes, memory_size):
    N = layer_sizes
    L = len(N)-1
    f, df, af = [np.tanh]*L, [mu.tanh_df]*L, [np.arctanh]*L
    f = {k: np.tanh for k in range(1,L+1)}
    df = {k: mu.tanh_df for k in range(1,L+1)}
    af = {k: np.arctanh for k in range(1,L+1)}
    for tries in range(10):
        sqn = BackPropSeqNet(N=N,f=f,df=df,af=af)
        sqn.pretrain(memory_size, eta=0.01, num_epochs=10000, term = 0.1, verbose=1)
        # mem check
        k = sqn.first()
        keys = np.empty((sqn.layer_size(), memory_size))
        keys[:,[0]] = k
        clobbered = np.zeros(memory_size, dtype=bool)
        for m in range(1,memory_size):
            k = sqn.next(k)
            keys[:,[m]] = k
            clobbered[:m] |= (keys[:,:m]==keys[:,[m]]).all(axis=0)
        seq_mem_check = (not clobbered.any())
        if seq_mem_check: break
    if not seq_mem_check: return False
    else: return sqn

def make_tanh_bmn(layer_size, memory_size, kv_layers=3, sq_layers=3):
    N_kv = [layer_size]*kv_layers
    L_kv = len(N_kv)-1
    f_kv = {k: np.tanh for k in range(1,L_kv+1)}
    df_kv = {k: mu.tanh_df for k in range(1,L_kv+1)}
    af_kv = {k: np.arctanh for k in range(1,L_kv+1)}
    N_sq = [layer_size]*sq_layers
    L_sq = len(N_sq)-1
    f_sq = {k: np.tanh for k in range(1,L_sq+1)}
    df_sq = {k: mu.tanh_df for k in range(1,L_sq+1)}
    af_sq = {k: np.arctanh for k in range(1,L_sq+1)}
    # retrain sequence until no clobbers
    for tries in range(10):
        bmn = BackPropMemoryNet(N_kv, f_kv, df_kv, af_kv, N_sq, f_sq, df_sq, af_sq, memory_size)
        bmn.train_sequence(eta=0.01, num_epochs=10000, term = 0.1, verbose=1)
        # mem check
        k = bmn.first()
        keys = np.empty((bmn.layer_size(), memory_size))
        keys[:,[0]] = k
        clobbered = np.zeros(memory_size, dtype=bool)
        for m in range(1,memory_size):
            k = bmn.next(k)
            keys[:,[m]] = k
            clobbered[:m] |= (keys[:,:m]==keys[:,[m]]).all(axis=0)
        seq_mem_check = (not clobbered.any())
        if seq_mem_check: break
    assert(seq_mem_check) # if not, in trouble
    return bmn

def kvnet_test():
    N = [32]*3
    L = len(N)-1
    f, df, af = [np.tanh]*L, [mu.tanh_df]*L, [np.arctanh]*L
    f = {k: np.tanh for k in range(1,L+1)}
    df = {k: mu.tanh_df for k in range(1,L+1)}
    af = {k: np.arctanh for k in range(1,L+1)}
    kvn = BackPropKVNet(N=N,f=f,df=df,af=af)
    # num_patterns = 5
    num_patterns = 1
    values = mu.random_patterns(N[0],num_patterns)
    keys = np.empty((N[0],num_patterns))
    # k = kvn.first()
    k = mu.random_patterns(N[0],1)
    print(k)
    for m in range(values.shape[1]):
        keys[:,[m]] = k
        kvn.write(k,values[:,[m]],verbose=1,term=0.5,eta=0.001,num_epochs=100000)
        # kvn.passive_ticks(10000)
        k = kvn.next(k)
    # memory accuracy
    net_values = kvn.read(keys)
    print('final accuracy:')
    print(np.fabs(values-net_values).max(axis=0))
    # show write histories
    # print('target vs read')
    # print(values[:,-1].T)
    # print(kvn.read(keys[:,-1]).T)
    h = [None,None]
    for c in range(kvn.cheat_error_history.shape[0]):
        h[0] = plt.plot(kvn.cheat_error_history[c,:],'r.')[0]
    h[1] = plt.plot(kvn.write_error_history,'b.')[0]
    # plt.plot(kvn.cheat_error_history.sum(axis=0) + kvn.write_error_history,'g.')
    # print(kvn.write_error_history)
    plt.xlabel('Training iteration')
    plt.ylabel('L_inf error on previous and current memories')
    plt.title('learning curves during write(k,v)')
    plt.legend(h,['||read(k_prev)-sign(read(k_prev))||','||read(k)-v||'])
    plt.show()

def sqnet_test():
    N = [32]*3
    L = len(N)-1
    f, df, af = [np.tanh]*L, [mu.tanh_df]*L, [np.arctanh]*L
    f = {k: np.tanh for k in range(1,L+1)}
    df = {k: mu.tanh_df for k in range(1,L+1)}
    af = {k: np.arctanh for k in range(1,L+1)}
    sqn = BackPropSeqNet(N=N,f=f,df=df,af=af)
    memory_size = 40
    sqn.pretrain(memory_size, eta=0.01, num_epochs=10000, term = 0.1, verbose=1)
    # mem check
    k = sqn.first()
    keys = np.empty((sqn.layer_size(), memory_size))
    keys[:,[0]] = k
    clobbered = np.zeros(memory_size, dtype=bool)
    for m in range(1,memory_size):
        k = sqn.next(k)
        keys[:,[m]] = k
        clobbered[:m] |= (keys[:,:m]==keys[:,[m]]).all(axis=0)
    seq_mem_check = (not clobbered.any())
    print('Final error: %f'%sqn.error_history[-1])
    print('mem check passed: %s'%str(seq_mem_check))
    print(clobbered)
    print(mu.patterns_to_ints(keys))
    print(mu.patterns_to_ints(keys)[:,np.newaxis] == mu.patterns_to_ints(keys)[np.newaxis,:])
    # learning curve
    plt.plot(sqn.error_history,'r.')
    plt.xlabel('Training iteration')
    plt.ylabel('L_inf error on key sequence')
    plt.title('sequence pretraining learning curve')
    plt.show()

def bmnet_test():
    num_patterns = 10
    net = make_tanh_bmn(layer_size = 32, memory_size=num_patterns*4, kv_layers=3, sq_layers=3)
    values = mu.random_patterns(net.layer_size(),num_patterns)
    keys = np.empty((net.layer_size(),num_patterns))
    k = net.first()
    for m in range(values.shape[1]):
        print('writing pattern %d'%m)
        keys[:,[m]] = k
        net.write(k,values[:,[m]],verbose=1,term=0.5,eta=0.01,num_epochs=10000)
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
    bmnet_test()
