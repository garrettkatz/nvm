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
        return self._noise(ut.int_to_pattern(self.N, ut.pattern_to_int(self.N,k)+1))
    def first(self):
        return self._noise(ut.int_to_pattern(self.N, 0))
    def passive_tick(self):
        pass

class MemoryNetTrace:
    """
    Wrap network to record key sequence and ideal key value mapping
    """
    def __init__(self, net):
        self.net = net
        self.key_sequence = {} # k -> next k
        self.key_value_map = {} # k -> value
    def read(self, k):
        return self.net.read(k)
    def write(self, k, v):
        self.net.write(k,v)
        self.key_value_map[ut.hash_pattern(k)] = v
        if len(self.key_value_map) >= 2**self.net.N:
            print('Warning: Address capacity reached!')
    def next(self, k):
        k_next = self.net.next(k)
        self.key_sequence[ut.hash_pattern(k)] = k_next
        return k_next
    def first(self):
        return self.net.first()
    def passive_tick(self):
        self.net.passive_tick()
    def memory_string(self, to_int=True):
        def _val(k):
            if ut.hash_pattern(k) in self.key_value_map:
                return self.key_value_map[ut.hash_pattern(k)]
            else: return np.nan*np.ones((self.net.N,1))
        k = self.first()
        keys = [k]
        values = [_val(k)]
        while ut.hash_pattern(k) in self.key_sequence:
            k = self.key_sequence[ut.hash_pattern(k)]
            keys.append(k)
            values.append(_val(k))
        if to_int:
            string = 'keys/values:\n'
            string += str(np.array([
                [ut.pattern_to_int(self.net.N,k) for k in keys],
                [ut.pattern_to_int(self.net.N,v) for v in values]]))
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

def kv_trial_data(N, num_mappings, max_passive_ticks):
    keys = np.sign(np.random.randn(N, num_mappings))
    values = np.sign(np.random.randn(N, num_mappings))
    num_passive_ticks = np.random.randint(max_passive_ticks, size=(num_mappings,))
    return keys, values, num_passive_ticks
    
def run_kv_trial(mnt, keys, values, num_passive_ticks):
    M = keys.shape[1]
    kv_accuracy = np.empty(M)
    for m in range(M):
        mnt.write(keys[:,[m]], values[:,[m]])
        for t in range(num_passive_ticks[m]):
            mnt.passive_tick()
        kv_accuracy[m] = mnt.key_value_accuracy()
    return kv_accuracy

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
    # mnt = MemoryNetTrace(mmn)
    # trial_data = kv_trial_data(N, num_mappings=10, max_passive_ticks=5)
    # acc = run_kv_trial(mnt, *trial_data)
    # print(acc)

    num_trials = 5
    trial_funs = [run_kv_trial] * num_trials
    trial_kwargs = []
    for n in range(num_trials):
        keys, values, num_passive_ticks = kv_trial_data(N, num_mappings=10, max_passive_ticks=5)
        trial_kwargs.append({
            'mnt': MemoryNetTrace(MockMemoryNet(N, noise=0.005)),
            'keys': keys,
            'values': values,
            'num_passive_ticks':num_passive_ticks,
        })

    num_procs = 2
    results = pool_trials(num_procs, trial_funs, trial_kwargs, results_file_name=None)
    for r in results:
        print(r)
