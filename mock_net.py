import numpy as np

def eq(pattern1, pattern2):
    return (np.sign(pattern1) == np.sign(pattern2)).all()
def cp(pattern):
    if type(pattern) == str: return pattern
    return pattern.copy()

class MockCoding:
    """
    Mapping between machine- and human-readable constants
    """
    def __init__(self, layer_size):
        self.layer_size = layer_size
        self.next_key = 0
        self.machine_readable_of = {}
        self.human_readable_of = {}
    def encode(self, human_readable):
        # return encoding if already encoded
        if human_readable in self.machine_readable_of:
            return self.machine_readable_of[human_readable]
        # encode with next key
        key = self.next_key
        self.next_key += 1
        # convert integer key to activity pattern
        machine_readable = np.random.rand(self.layer_size)
        for unit in range(self.layer_size):
            machine_readable[unit] *= (-1)**(1 & (key >> unit))
        # save encoding (using hashable bytes) and return
        self.machine_readable_of[human_readable] = machine_readable
        self.human_readable_of[machine_readable.tobytes()] = human_readable
        return machine_readable
    def decode(self, machine_readable):
        # if not already encoded return unknown string
        if machine_readable.tobytes() not in self.human_readable_of:
            return '<?>'
        # else return decoding
        return self.human_readable_of[machine_readable.tobytes()]

class MockNet:
    """
    Patterns: numpy arrays
    Pattern hashes: pattern_hash[layer_name] == pattern
    Representing dynamical state transitions:
    transitions[layer_name] = [(old_pattern_hash,new_pattern),...]
    in the presense of the old patterns, the layer will transition to the new pattern
    rules earlier in list take precedence
    "patterns" in rules can be strings, in which case they are treated as variables bound at activation time
    """
    def __init__(self, layer_names, layer_sizes, history=2):
        self.layer_names = layer_names + ['A','W']
        self.history = history
        self.tick_mark = 0
        self.layers = {}
        self.transitions = {}
        # gates
        self.gate_index_map = {}
        index = 0
        for to_layer_name in layer_names:
            for from_layer_name in layer_names:
                self.gate_index_map[to_layer_name, from_layer_name] = index
                index += 1
        layer_sizes += [len(layer_names)**2]*2
        for (layer_name, layer_size) in zip(self.layer_names, layer_sizes):
            self.layers[layer_name] = -np.ones((history, layer_size))
            self.transitions[layer_name] = [({layer_name:layer_name}, layer_name)]
    def get_layer_names(self):
        return list(self.layer_names)
    def get_pattern(self, layer_name, tick_offset=0):
        tick_mark = (self.tick_mark + tick_offset) % self.history
        return self.layers[layer_name][tick_mark].copy()
    def set_pattern(self, layer_name, layer_pattern, tick_offset=0):
        tick_mark = (self.tick_mark + tick_offset) % self.history
        self.layers[layer_name][tick_mark] = layer_pattern.copy()
    def get_patterns(self):
        return {layer_name: self.get_pattern(layer_name) for layer_name in self.layer_names}
    def set_patterns(self, pattern_hash):
        for layer_name in self.layer_names:
            self.set_pattern(layer_name, pattern_hash[layer_name])
    def list_patterns(self):
        return [(layer_name, self.get_pattern(layer_name)) for layer_name in self.layer_names]
    def get_gates(self):
        gate_hash = {}
        for gate_layer_name in ['A','W']:
            pattern = self.get_pattern(gate_layer_name)
            for to_layer_name, from_layer_name in self.gate_index_map:
                index = self.gate_index_map[to_layer_name, from_layer_name]
                gate_hash[gate_layer_name, to_layer_name, from_layer_name] = activity_gates[index]
        return gate_hash
    def get_gate_index(self, to_layer_name, from_layer_name):
        return self.gate_index_map[to_layer_name, from_layer_name]
    def activate(self, layer_name, old_pattern_hash):
        # check transitions
        for pattern_hash, new_pattern in self.transitions[layer_name]:
            matches = True
            pattern_vars = {}
            # check for a match
            for layer_name in pattern_hash:
                pattern = pattern_hash[layer_name]
                if type(pattern) == str: # pattern variable
                    if pattern not in pattern_vars: # enforces consistency
                        pattern_vars[pattern] = old_pattern_hash[layer_name]
                    pattern = pattern_vars[pattern]
                if not eq(pattern, old_pattern_hash[layer_name]):
                    matches = False
                    break
            # apply a match
            if matches:
                # print('rule fired:')
                # print((pattern_hash, new_pattern))
                if type(new_pattern) == str: # pattern variable
                    new_pattern = pattern_vars[new_pattern]
                break
        if matches:
            new_pattern = new_pattern.copy()
        else:
            print('no rule fired')
            new_pattern = self.get_pattern(layer_name)
        return new_pattern.copy()
    def advance_tick_mark(self):
        self.tick_mark += 1
        self.tick_mark %= self.history
    def tick(self):
        # get new layer activations
        old_pattern_hash = self.get_patterns()
        new_pattern_hash = {}
        for layer_name in self.layers:
            new_pattern_hash[layer_name] = self.activate(layer_name, old_pattern_hash)
        # update network
        self.advance_tick_mark()
        self.set_patterns(new_pattern_hash)
        # learn associations
        pass
    def train(self, pattern_hash, new_pattern_hash):
        pattern_hash = {layer_name: cp(pattern_hash[layer_name]) for layer_name in pattern_hash}
        for layer_name in new_pattern_hash:
            new_pattern = cp(new_pattern_hash[layer_name])
            self.transitions[layer_name].insert(0,(pattern_hash, new_pattern))
    
if __name__ == '__main__':
    layer_size = 4
    net = MockNet(['A','B','C'], [layer_size]*3)
    ones = np.ones(layer_size)
    print(net.layer_names)
    old_pattern_hash = {'A':ones,'B':ones,'C':ones}
    print('activate:')
    print(net.activate('A',old_pattern_hash))

    # mvm = make_nvm_mocknet(3)
    # print(mvm.layer_names)

# class MockNet:
#     def __init__(self, num_registers, layer_size=16, io_modules=[]):
#         register_names = ['{%d}'%r for r in range(num_registers)]
#         self.layer_size = layer_size
#         self.module_names = []
#         self.modules = {}
#         self.layer_module_map = {}
#         self.add_module(MockModule('control',['IP','OPC','OP1','OP2','OP3'] + register_names, layer_size))
#         self.add_module(MockModule('compare',['C1','C2','CO'],layer_size))
#         self.add_module(MockModule('nand',['N1','N2','NO'],layer_size))
#         self.add_module(MockMemoryModule(layer_size))
#         for io_module in io_modules:
#             self.add_module(io_module)
#         self.add_module(MockGatingModule(self.get_layer_names()))
#     def add_module(self, module):
#         self.module_names.append(module.module_name)
#         self.modules[module.module_name] = module
#         for layer_name in module.layer_names:
#             self.layer_module_map[layer_name] = module.module_name
#     def get_module(self, module_name):
#         return self.modules[module_name]
#     def get_layer_names(self):
#         layer_names = []
#         for name in self.module_names:
#             layer_names += self.modules[name].layer_names
#         return layer_names
#     def get_pattern(self, layer_name):
#         module_name = self.layer_module_map[layer_name]
#         return self.modules[module_name].get_pattern(layer_name)
#     def list_patterns(self):
#         pattern_list = []
#         for module_name in self.module_names:
#             pattern_list += self.modules[module_name].list_patterns()
#         return pattern_list
#     def hash_patterns(self):
#         return dict(self.list_patterns())
#     def set_pattern(self,layer_name, pattern):
#         module_name = self.layer_module_map[layer_name]
#         self.modules[module_name].set_pattern(layer_name, pattern)
#     def set_patterns(self, pattern_hash):
#         for layer_name in pattern_hash:
#             self.set_pattern(layer_name, pattern_hash[layer_name])
#     def tick(self):
#         old_pattern_hash = self.hash_patterns()
#         new_pattern_hash = {}
#         # activate modules
#         for module_name in self.modules:
#             new_pattern_hash.update(self.modules[module_name].activate(old_pattern_hash))
#         # layer copies
#         for to_layer_name in self.modules['gating'].net_layer_names:
#             for from_layer_name in self.modules['gating'].net_layer_names:
#                 if self.modules['gating'].get_gate('A',to_layer_name, from_layer_name) > 0.5:
#                     new_pattern_hash[to_layer_name] =  old_pattern_hash[from_layer_name]
#         # update network
#         for module_name in self.modules:
#             self.modules[module_name].advance_tick_mark()
#             self.modules[module_name].set_hashed_patterns(new_pattern_hash)
#         # associations
        
# class MockModule:
#     def __init__(self, module_name, layer_names, layer_size=16, history=2):
#         self.module_name = module_name
#         self.layer_names = layer_names
#         self.layer_size = layer_size
#         self.history = history
#         self.layers = {name: -np.ones((history, layer_size)) for name in layer_names}
#         self.tick_mark = 0
#         # default transition: sustain activity
#         pattern_list = zip(self.layer_names,self.layer_names)
#         self.transitions = [(pattern_list, pattern_list)]
#     def get_pattern(self, layer_name, tick_offset=0):
#         tick_mark = (self.tick_mark + tick_offset) % self.history
#         return self.layers[layer_name][tick_mark].copy()
#     def set_pattern(self, layer_name, layer_pattern):
#         self.layers[layer_name][self.tick_mark] = layer_pattern.copy()
#     def list_patterns(self):
#         return [(layer_name, self.get_pattern(layer_name)) for layer_name in self.layer_names]
#     def hash_patterns(self):
#         return {layer_name: self.get_pattern(layer_name) for layer_name in self.layer_names}
#     def set_hashed_patterns(self, pattern_hash):
#         for layer_name in self.layer_names:
#             self.set_pattern(layer_name, pattern_hash[layer_name])
#     def advance_tick_mark(self):
#         self.tick_mark += 1
#         self.tick_mark %= self.history
#     def activate(self, old_pattern_hash):
#         # persist if no transitions apply
#         new_pattern_hash = {layer_name:cp(old_pattern_hash[layer_name]) for layer_name in old_pattern_hash}
#         # check transitions
#         for pattern_list, next_pattern_list in self.transitions:
#             matches = True
#             pattern_vars = {}
#             # check for a match
#             for (layer_name, pattern) in pattern_list:
#                 if type(pattern) == str: # pattern variable
#                     if pattern not in pattern_vars: # enforces consistency
#                         pattern_vars[pattern] = old_pattern_hash[layer_name]
#                     pattern = pattern_vars[pattern]
#                 if not eq(pattern, old_pattern_hash[layer_name]):
#                     matches = False
#                     break
#             # apply a match
#             if matches:
#                 print('%s rule fired:'%self.module_name)
#                 print((pattern_list, next_pattern_list))
#                 for (layer_name, pattern) in next_pattern_list:
#                     if type(pattern) == str: # pattern variable
#                         pattern = pattern_vars[pattern]
#                     new_pattern_hash[layer_name] = pattern
#                 break
#         if not matches: print('%s: no rule fired'%self.module_name)
#         return new_pattern_hash
#     def train(self, pattern_list, next_pattern_list):
#         """
#         train the module dynamics on the given transition.
#         both pattern_lists should include patterns for every layer in this module.
#         after training, when presented with the same patterns in pattern_list,
#         the module layers will transition to the same patterns in next_pattern_list.
#         more newly learned transitions take precedence.
#         if a presented pattern is a string, it is treated as a variable that gets bound at activation time.
#         each occurrence of the same pattern variable is constrained to the same value when bound.
#         """        
#         pattern_list = [(layer_name, cp(pattern)) for (layer_name, pattern) in pattern_list]
#         next_pattern_list = [(layer_name, cp(pattern)) for (layer_name, pattern) in next_pattern_list]
#         self.transitions.insert(0,(pattern_list, next_pattern_list))
#     def associate(self, gate_hash):
#         """
#         Associate current and previous activity patterns within module
#         """
#         current_pattern_list, previous_pattern_list = [], []
#         for to_layer_name in self.layer_names:
#             if any([gate_hash['W',to_layer_name,from_layer_name] > .5 for from_layer_name in self.layer_names]):
#                 current_pattern_list.append((to_layer_name, self.get_pattern(to_layer_name)))
#             else:
#                 current_pattern_list.append((to_layer_name, to_layer_name))
#         for from_layer_name in self.layer_names:
#             if any([gate_hash['W',to_layer_name,from_layer_name] > .5 for to_layer_name in self.layer_names]):
#                 previous_pattern_list.append((from_layer_name, self.get_pattern(from_layer_name)))
#             else:
#                 previous_pattern_list.append((from_layer_name, from_layer_name))
#         self.train(previous_pattern_list, current_pattern_list)

# if __name__ == '__main__':

#     # Association
#     mod = MockModule(module_name='mod', layer_names=['U','V'], layer_size=4)
#     ones = np.ones(mod.layer_size)
#     mod.set_hashed_patterns({'U':ones, 'V':ones})
#     mod.advance_tick_mark()
#     mod.set_hashed_patterns({'U':ones, 'V':-ones})
#     gate_hash = {('W','U','U'):0.0,('W','U','V'):0.0,('W','V','U'):1.0,('W','V','V'):1.0}
#     mod.associate(gate_hash)
#     print('transitions:')
#     print(mod.transitions)
#     old_hash = {'U':ones, 'V':ones}
#     new_hash = mod.activate(old_hash)
#     mod.advance_tick_mark()
#     print('newhash:')
#     print(new_hash)

#     # # Training
#     # mod = MockModule(module_name='mod', layer_names=['V'], layer_size=4)
#     # ones = np.ones(mod.layer_size)
#     # transitions = [
#     #     [[('V', -ones),('U',+ones)],[('V', +ones)]],
#     #     [[('V', +ones),('U',+ones)],[('V', -ones)]],
#     #     [[('U',-ones)],[('V', -ones)]],
#     # ]
#     # for pattern_list, next_pattern_list in transitions:
#     #     mod.train(pattern_list, next_pattern_list)
#     # print('flashed')
#     # mod.set_pattern('V',-ones)
#     # pattern_hash = {'V':-ones, 'U': +ones}
#     # new_hash = mod.activate(pattern_hash)
#     # mod.advance_tick_mark()
#     # mod.set_hashed_patterns(new_hash)
#     # print(mod.get_pattern('V'))
#     # pattern_hash = {'V':+ones, 'U': +ones}
#     # new_hash = mod.activate(pattern_hash)
#     # mod.advance_tick_mark()
#     # mod.set_hashed_patterns(new_hash)
#     # print(mod.get_pattern('V'))
#     # pattern_hash = {'V':-ones, 'U': +ones}
#     # new_hash = mod.activate(pattern_hash)
#     # mod.advance_tick_mark()
#     # mod.set_hashed_patterns(new_hash)
#     # print(mod.get_pattern('V'))
#     # pattern_hash = {'V':+ones, 'U': -ones}
#     # new_hash = mod.activate(pattern_hash)
#     # mod.advance_tick_mark()
#     # mod.set_hashed_patterns(new_hash)
#     # print(mod.get_pattern('V'))
#     # pattern_hash = {'V':-ones, 'U': np.array([1,-1,1,-1])}
#     # new_hash = mod.activate(pattern_hash)
#     # mod.advance_tick_mark()
#     # mod.set_hashed_patterns(new_hash)
#     # print(mod.get_pattern('V'))

# class MockGatingModule(MockModule):
#     def __init__(self, net_layer_names):
#         MockModule.__init__(self, module_name='gating', layer_names=['A','W'], layer_size=len(net_layer_names)**2)
#         self.net_layer_names = net_layer_names
#         self.gate_index_map = {}
#         index = 0
#         for to_layer_name in self.net_layer_names:
#             for from_layer_name in self.net_layer_names:
#                 self.gate_index_map[to_layer_name, from_layer_name] = index
#                 index += 1
#     def get_gate(self, gate_layer_name, to_layer_name, from_layer_name):
#         node_index = self.gate_index_map[to_layer_name,from_layer_name]
#         return self.layers[gate_layer_name][self.tick_mark, node_index]
#     def hash_gates(self):
#         gate_hash = {}
#         activity_gates = self.hash_patterns('A')
#         weight_gates = self.hash_patterns('W')
#         for to_layer_name in self.net_layer_names:
#             for from_layer_name in self.net_layer_names:
#                 index = self.gate_index_map[to_layer_name, from_layer_name]
#                 pattern = self.get_pattern('A')
#                 gate_hash['A', to_layer_name, from_layer_name] = activity_gates[index]
#                 gate_hash['W', to_layer_name, from_layer_name] = weight_gates[index]
#         return gate_hash

# class MockMemoryModule(MockModule):
#     def __init__(self, layer_size=16):
#         MockModule.__init__(self, module_name='memory', layer_names=['K','V'], layer_size=layer_size)
#     def activate(self, old_pattern_hash):
#         new_pattern_hash = {}
#         for layer_name in self.layer_names:
#             new_pattern = np.tanh(np.random.randn(self.layer_size))
#             new_pattern_hash[layer_name] = new_pattern
#         return new_pattern_hash

# class MockIOModule(MockModule):
#     def __init__(self, module_name, layer_size=16):
#         MockModule.__init__(self, module_name, layer_names=['STDI','STDO'], layer_size=layer_size)
