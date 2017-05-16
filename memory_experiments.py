import multiprocessing as mp
import pickle as pkl
import time as time
import numpy as np
import matplotlib.pyplot as plt
import memory_utils as mu
import backprop_memory_net as bmn

np.set_printoptions(linewidth=1000)

def save_pkl_file(filename, data):
    """
    Convenience function for pickling data to a file
    """
    pkl_file = open(filename,'w')
    pkl.dump(data, pkl_file)
    pkl_file.close()
def load_pkl_file(filename):
    """
    Convenience function for loading pickled data from a file
    """
    pkl_file = open(filename,'r')
    data = pkl.load(pkl_file)
    pkl_file.close()
    return data
def save_npz_file(filename, **kwargs):
    """
    Convenience function for saving numpy data to a file
    Each kwarg should have the form
      array_name=array
    """
    npz_file = open(filename,"w")
    np.savez(npz_file, **kwargs)
    npz_file.close()
def load_npz_file(filename):
    """
    Convenience function for loading numpy data from a file
    returns npz, a dictionary with key-value pairs of the form
      array_name: array
    """    
    npz = np.load(filename)
    npz = {k:npz[k] for k in npz.files}
    return npz

class MockMemoryNet:
    def __init__(self, N, noise=0):
        self.N = N
        self.noise = noise
        self.key_value_map = {} # key-value "network"
    def _noise(self,p):
        return p * (-1)**(np.random.rand(self.N,1) < self.noise)
    def layer_size(self):
        return self.N
    def read(self,k):
        if mu.hash_pattern(k) in self.key_value_map:
            return self.key_value_map[mu.hash_pattern(k)]
        else:
            return np.sign(np.random.randn(self.N,1))
    def write(self,k,v):
        self.key_value_map[mu.hash_pattern(self._noise(k))] = self._noise(v)
    def next(self, k):
        return self._noise(mu.int_to_pattern(self.N, mu.patterns_to_ints(k)[0]+1))
    def first(self):
        return self._noise(mu.int_to_pattern(self.N, 0))
    def passive_ticks(self, num_ticks):
        pass

class MemoryNetHarness:
    """
    Wrap network to record key sequence and ideal key value mapping
    """
    def __init__(self, net, passive_tick_fun=None, num_passive_ticks=0, seed=None):
        # passive_tick_fun(random_state) returns a random number of passive ticks
        # if passive_tick_fun == None: passive_tick_fun = mu.constant_tick_fun(0)
        if seed == None: seed = np.random.randint(2**32)
        self.net = net
        self.passive_tick_fun = passive_tick_fun
        self.num_passive_ticks = num_passive_ticks
        self.random_state = np.random.RandomState(seed)
        self.key_sequence = {} # k -> next k
        self.key_value_map = {} # k -> value
    def random_patterns(self, num_patterns):
        return np.sign(self.random_state.randn(self.net.layer_size(), num_patterns))
    def passive_ticks(self):
        # pending fix for compatibility with Pool
        num_ticks = mu.constant_tick_fun(self.num_passive_ticks)(self.random_state)
        #num_ticks = self.passive_tick_fun(self.random_state)
        self.net.passive_ticks(num_ticks)
    def read(self, k):
        v = self.net.read(k)
        self.passive_ticks()
        return v
    def write(self, k, v, num_epochs=None):
        if num_epochs is None: self.net.write(k,v)
        else: self.net.write(k,v,num_epochs=num_epochs)
        self.key_value_map[mu.hash_pattern(k)] = v
        if len(self.key_value_map) >= 2**self.net.layer_size():
            print('Warning: Address capacity reached!')
        self.passive_ticks()        
    def next(self, k):
        k_next = self.net.next(k)
        self.key_sequence[mu.hash_pattern(k)] = k_next
        self.passive_ticks()
        return k_next
    def first(self):
        return self.net.first()
    def memory_string(self, to_int=True):
        def _val(k):
            if mu.hash_pattern(k) in self.key_value_map:
                return self.key_value_map[mu.hash_pattern(k)]
            else: return np.nan*np.ones((self.net.layer_size(),1))
        # key sequence
        k = self.first()
        keys = [k]
        values = [_val(k)]
        while mu.hash_pattern(k) in self.key_sequence:
            k = self.key_sequence[mu.hash_pattern(k)]
            keys.append(k)
            values.append(_val(k))
        if to_int:
            string = 'keys/values:\n'
            string += str(np.concatenate([
                mu.patterns_to_ints(keys),
                mu.patterns_to_ints(values)],axis=0))
        else:
            string = 'keys:\n'
            string += str(np.concatenate(keys,axis=1))
            string += '\nvalues:\n'
            string += str(np.concatenate(values,axis=1))
        return string
    def sequence_accuracy(self):
        keys = self.key_sequence.keys()
        test_sequence = np.concatenate([self.key_sequence[k] for k in keys],axis=1)
        net_sequence = np.concatenate([self.net.next(mu.unhash_pattern(k)) for k in keys],axis=1)
        #acc = lambda x: 1.0*x.sum()/x.size
        acc = lambda x: 1.0*x.all(axis=0).sum()/x.shape[1]
        return acc(test_sequence==net_sequence)
    def key_value_accuracy(self):
        keys = self.key_value_map.keys()
        test_map = np.concatenate([self.key_value_map[k] for k in keys],axis=1)
        net_map = np.concatenate([self.net.read(mu.unhash_pattern(k)) for k in keys],axis=1)
        #acc = lambda x: 1.0*x.sum()/x.size
        acc = lambda x: 1.0*x.all(axis=0).sum()/x.shape[1]
        return acc(test_map==net_map)

def run_array_write_trial(mnh, values, write_epochs, params, verbose=0):
    N, M = values.shape
    kv_accuracy, seq_accuracy = np.empty(M), np.empty(M)
    keys = np.empty((N, M))
    key = mnh.first()
    for m in range(M):
        keys[:,[m]] = key
        mnh.write(key, values[:,[m]], num_epochs=write_epochs)
        key = mnh.next(key)
        kv_accuracy[m] = mnh.key_value_accuracy()
        seq_accuracy[m] = mnh.sequence_accuracy()
    # sanity check
    net_keys = np.empty(keys.shape)
    net_values = np.empty(values.shape)
    for m in range(M):
        net_keys[:,[m]] = mnh.next(keys[:,[m]])
        net_values[:,[m]] = mnh.read(keys[:,[m]])
    clobbered = np.zeros(keys.shape[1],dtype=bool)
    for m in range(M):
        clobbered[:m] |= (keys[:,:m]==keys[:,[m]]).all(axis=0)
    seq_mem_check = (not clobbered.any()) and (net_keys[:,:-1]==keys[:,1:]).all()
    kv_mem_check = (values[:,~clobbered]==net_values[:,~clobbered]).all()
    if verbose > 0:
        print(params)
    if verbose > 1:
        print('***kv trial***')
        print('clobbered:')
        print(clobbered)
        print('seq keys:')
        print(mu.patterns_to_ints(keys[:,1:]))
        print(mu.patterns_to_ints(net_keys[:,:-1]))
        print('kv values:')
        print(mu.patterns_to_ints(keys))
        print(mu.patterns_to_ints(values))
        print(mu.patterns_to_ints(net_values))
        print('seq, kv mem check:')
        print(seq_mem_check, kv_mem_check)
        print('seq, kv acc:')
        print(seq_accuracy)
        print(kv_accuracy)
        print(seq_accuracy[-1], kv_accuracy[-1])
        print('%d failed writes'%mnh.net.failed_write_count)
        # h = [None,None]
        # for c in range(mnh.net.cheat_error_history.shape[0]):
        #     h[0] = plt.plot(mnh.net.cheat_error_history[c,:],'r.')[0]
        # h[1] = plt.plot(mnh.net.write_error_history,'b.')[0]
        # plt.xlabel('Training iteration')
        # plt.ylabel('L_inf error on previous and current memories')
        # plt.title('learning curves during write(k,v)')
        # plt.legend(h,['||read(k_prev)-sign(read(k_prev))||','||read(k)-v||'])
        # plt.show()
    return seq_accuracy, kv_accuracy, seq_mem_check, kv_mem_check

def run_array_rotate_trial(mnh, values):
    M = values.shape[1]
    kv_accuracy = []
    # write values to array
    k = mnh.first()
    for m in range(M):
        mnh.write(k, values[:,[m]])
        kv_accuracy.append(mnh.key_value_accuracy())
        k = mnh.next(k)
    # rotate array
    k = mnh.first()
    v_prev = mnh.read(k)
    for m in range(1,M):
        k = mnh.next(k)
        v = mnh.read(k)
        mnh.write(k, v_prev)
        kv_accuracy.append(mnh.key_value_accuracy())
        v_prev = v
    mnh.write(mnh.first(), v_prev)
    kv_accuracy.append(mnh.key_value_accuracy())
    # sanity check
    net_values = []
    k = mnh.first()
    for m in range(M):
        net_values.append(mnh.read(k))
        k = mnh.next(k)
    net_values = np.concatenate(net_values,axis=1)
    mem_check = (values.shape == net_values.shape) and (np.roll(values,1,axis=1)==net_values).all()
    print('***array trial***')
    print('array rotate:')
    print(mu.patterns_to_ints(values))
    print(mu.patterns_to_ints(net_values))
    print('mem check:')
    print(mem_check)
    print('acc:')
    print(kv_accuracy)
    return kv_accuracy, mem_check

def run_pooled_trial(trial_fun_and_kwargs):
    trial_fun, trial_kwargs = trial_fun_and_kwargs
    return trial_fun(**trial_kwargs)

def pool_trials(num_procs, trial_funs, trial_kwargs, results_file_name=None):
    # run each trial_funs[t](**trial_kwargs[t])
    # use num_procs processes
    # save returned list in results_file_name if not None
    
    cpu_count = mp.cpu_count()
    print('%d cpus, using %d'%(cpu_count, num_procs))

    pool_args = zip(trial_funs, trial_kwargs)
    start_time = time.time()
    if num_procs < 1: # don't multiprocess
        pool_results = [run_pooled_trial(args) for args in pool_args]
    else:
        pool = mp.Pool(processes=num_procs)
        pool_results = pool.map(run_pooled_trial, pool_args)
        pool.close()
        pool.join()
    print('total time: %f. saving results...'%(time.time()-start_time))

    if results_file_name is not None:
        results_file = open(results_file_name, 'w')
        pkl.dump(pool_results, results_file)
        results_file.close()

    return pool_results

def pooled_array_write_trials():
    # simple kv experiment:
    # learn a random sequence of kv mappings.
    # At each one, assess recall on all learned so far.
    # Between each one, allow some number of random passive ticks.

    # N = 8
    # mmn = MockMemoryNet(N, noise=0.005)
    # mnh = MemoryNetHarness(mmn)
    # trial_data = kv_trial_data(N, num_mappings=10, max_passive_ticks=5)
    # acc = run_kv_trial(mnh, *trial_data)
    # print(acc)

    run_trial_fun = run_array_write_trial
    # run_trial_fun = run_array_trial

    num_procs = 9
    array_length_grid = [4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,65]
    write_epochs_grid = [500,1000,5000,10000,50000]
    layer_size_grid = [32]
    num_passive_ticks=0

    num_trials = len(array_length_grid)*len(layer_size_grid)*len(write_epochs_grid)
    trial_funs = [run_trial_fun] * num_trials
    trial_kwargs = []
    seed = None

    params = []
    for array_length in array_length_grid:
        for layer_size in layer_size_grid:
            values = mu.random_patterns(layer_size, num_patterns=array_length)
            for write_epochs in write_epochs_grid:
                params.append((array_length,layer_size, write_epochs))
                net = bmn.make_tanh_bmn(layer_size, memory_size=array_length, kv_layers=3, sq_layers=3)
                # passive_tick_fun = mu.constant_tick_fun(num_passive_ticks)
                # mnh = MemoryNetHarness(net, passive_tick_fun=passive_tick_fun, seed=seed)
                mnh = MemoryNetHarness(net, num_passive_ticks=num_passive_ticks, seed=seed)
                trial_kwargs.append({
                    'mnh': mnh,
                    'values': values,
                    'write_epochs':write_epochs,
                    'params': params[-1],
                    'verbose': 1,
                })

    results = pool_trials(num_procs, trial_funs, trial_kwargs, results_file_name=None)

    print('array_length;layer_size;write_epochs')
    print(np.array(params).T)
    print('seq_acc, kv_acc, seq_mem, kv_mem')
    for idx in range(4):
        if idx < 2: print([r[idx][-1] for r in results])
        else: print([r[idx] for r in results])

    # confirm perfect accuracy iff memory check passes
    for r in results:
        assert((r[0][-1]==1. and r[1][-1]==1.) == (r[2] and r[3]))

    # save results
    save_pkl_file('bmn.pkl',{'params':params, 'results':results})

def show_array_write_results():
    data = load_pkl_file('bmn.pkl')
    params = np.array(data['params'])
    results = np.array(data['results'])
    plot_res = np.concatenate((params, np.array([[r[1][-1] for r in results]]).T),axis=1)
    print(plot_res)
    h, leg = [], []
    for epochs in np.unique(params[:,2]):
        idx = (params[:,2] == epochs)
        h.append(plt.plot(params[idx,0],plot_res[idx,3])[0])
        leg.append('write iters=%d'%(epochs))
        print(epochs)
    plt.legend(h,leg,loc='lower left')
    plt.xlabel('array length')
    plt.ylabel('# entries recalled')
    plt.show()
    

if __name__ == '__main__':
    # seed = 1234
    # mnh = MemoryNetHarness(MockMemoryNet(3, noise=0.005),seed=seed)
    # print(mnh.random_patterns(num_patterns=8))
    pooled_array_write_trials()
    show_array_write_results()
