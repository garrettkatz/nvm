import multiprocessing as mp
import pickle as pkl
import time as time
import numpy as np
import utils as ut

np.set_printoptions(linewidth=1000)

class MockMemoryNet:
    def __init__(self, N, noise=0):
        self.N = N
        self.noise = noise
        self.key_value_map = {} # key-value "network"
    def _noise(self,p):
        return p * (-1)**(np.random.rand(self.N,1) < self.noise)
    def read(self,k):
        if ut.hash_pattern(k) in self.key_value_map:
            return self.key_value_map[ut.hash_pattern(k)]
        else:
            return np.sign(np.random.randn(self.N,1))
    def write(self,k,v):
        self.key_value_map[ut.hash_pattern(self._noise(k))] = self._noise(v)
    def next(self, k):
        return self._noise(ut.int_to_pattern(self.N, ut.patterns_to_ints(k)[0]+1))
    def first(self):
        return self._noise(ut.int_to_pattern(self.N, 0))
    def passive_ticks(self, num_ticks):
        pass

class MemoryNetHarness:
    """
    Wrap network to record key sequence and ideal key value mapping
    """
    def __init__(self, net, max_passive_ticks=2, seed=None):
        if seed == None: seed = np.random.randint(2**32)
        self.net = net
        self.max_passive_ticks = max_passive_ticks
        self.rs = np.random.RandomState(seed)
        self.key_sequence = {} # k -> next k
        self.key_value_map = {} # k -> value
    def passive_ticks(self):
        num_ticks = self.rs.randint(self.max_passive_ticks+1)
        self.net.passive_ticks(num_ticks)
    def read(self, k):
        v = self.net.read(k)
        self.passive_ticks()
        return v
    def write(self, k, v):
        self.net.write(k,v)
        self.key_value_map[ut.hash_pattern(k)] = v
        if len(self.key_value_map) >= 2**self.net.N:
            print('Warning: Address capacity reached!')
        self.passive_ticks()        
    def next(self, k):
        k_next = self.net.next(k)
        self.key_sequence[ut.hash_pattern(k)] = k_next
        self.passive_ticks()
        return k_next
    def first(self):
        return self.net.first()
    def memory_string(self, to_int=True):
        def _val(k):
            if ut.hash_pattern(k) in self.key_value_map:
                return self.key_value_map[ut.hash_pattern(k)]
            else: return np.nan*np.ones((self.net.N,1))
        # key sequence
        k = self.first()
        keys = [k]
        values = [_val(k)]
        while ut.hash_pattern(k) in self.key_sequence:
            k = self.key_sequence[ut.hash_pattern(k)]
            keys.append(k)
            values.append(_val(k))
        if to_int:
            string = 'keys/values:\n'
            string += str(np.concatenate([
                ut.patterns_to_ints(keys),
                ut.patterns_to_ints(values)],axis=0))
        else:
            string = 'keys:\n'
            string += str(np.concatenate(keys,axis=1))
            string += '\nvalues:\n'
            string += str(np.concatenate(values,axis=1))
        return string
    def sequence_accuracy(self):
        keys = self.key_sequence.keys()
        test_sequence = np.concatenate([self.key_sequence[k] for k in keys],axis=1)
        net_sequence = np.concatenate([self.net.next(ut.unhash_pattern(k)) for k in keys],axis=1)
        acc = lambda x: 1.0*x.sum()/x.size
        return acc(test_sequence==net_sequence)
    def key_value_accuracy(self):
        keys = self.key_value_map.keys()
        test_map = np.concatenate([self.key_value_map[k] for k in keys],axis=1)
        net_map = np.concatenate([self.net.read(ut.unhash_pattern(k)) for k in keys],axis=1)
        acc = lambda x: 1.0*x.sum()/x.size
        return acc(test_map==net_map)

def kv_trial_data(N, num_mappings):
    keys = np.sign(np.random.randn(N, num_mappings))
    values = np.sign(np.random.randn(N, num_mappings))
    return keys, values
    
def run_kv_trial(mnh, keys, values):
    M = keys.shape[1]
    kv_accuracy = np.empty(M)
    for m in range(M):
        mnh.write(keys[:,[m]], values[:,[m]])
        kv_accuracy[m] = mnh.key_value_accuracy()
    # sanity check
    net_values = np.empty(values.shape)
    clobbered = np.zeros(keys.shape[1],dtype=bool)
    for m in range(M):
        clobbered[:m] |= (keys[:,:m]==keys[:,[m]]).all(axis=0)
        net_values[:,[m]] = mnh.read(keys[:,[m]])
    mem_check = (values[:,~clobbered]==net_values[:,~clobbered]).all()
    print('***kv trial***')
    print('clobbered:')
    print(clobbered)
    print(keys)
    print('kv values:')
    print(ut.patterns_to_ints(keys))
    print(ut.patterns_to_ints(values))
    print(ut.patterns_to_ints(net_values))
    # print('harness memory:')
    # print(mnh.memory_string())
    print('mem check:')
    print(mem_check)
    print('acc:')
    print(kv_accuracy[-1])
    return kv_accuracy, mem_check

def array_trial_data(N, array_length):
    values = np.sign(np.random.randn(N, array_length))
    return values
    
def run_array_trial(mnh, values):
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
    print(ut.patterns_to_ints(values))
    print(ut.patterns_to_ints(net_values))
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

if __name__ == '__main__':
    # simple kv experiment:
    # learn a random sequence of kv mappings.
    # At each one, assess recall on all learned so far.
    # Between each one, allow some number of random passive ticks.

    N = 8
    # mmn = MockMemoryNet(N, noise=0.005)
    # mnh = MemoryNetHarness(mmn)
    # trial_data = kv_trial_data(N, num_mappings=10, max_passive_ticks=5)
    # acc = run_kv_trial(mnh, *trial_data)
    # print(acc)

    num_trials = 5
    max_passive_ticks=5
    trial_funs = [run_kv_trial] * num_trials
    # trial_funs = [run_array_trial] * num_trials
    trial_kwargs = []
    seed = None
    for n in range(num_trials):
        keys, values = kv_trial_data(N, num_mappings=10)
        # values = array_trial_data(N, array_length=10)
        trial_kwargs.append({
            'mnh': MemoryNetHarness(MockMemoryNet(N, noise=0.00), max_passive_ticks=max_passive_ticks, seed=seed),
            'keys': keys,
            'values': values,
        })

    num_procs = 0
    results = pool_trials(num_procs, trial_funs, trial_kwargs, results_file_name=None)
    print([mem_check for (_,mem_check) in results])
    print([acc[-1] for (acc,_) in results])
