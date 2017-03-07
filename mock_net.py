import numpy as np

def eq(pattern1, pattern2):
    return (np.sign(pattern1) == np.sign(pattern2)).all()

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
    def __init__(self, mock_coding, num_registers, layer_size=16, io_modules=[]):
        self.num_registers = num_registers
        self.register_names = ['IP','OPC','OP1','OP2','OP3'] + ['{%d}'%r for r in range(num_registers)]
        self.layer_size = layer_size
        self.module_names = []
        self.modules = {}
        for register_name in self.register_names: mock_coding.encode(register_name)
        self.add_module(MockModule('control',self.register_names, layer_size))
        self.add_module(MockModule('compare',['C1','C2','CO'],layer_size))
        self.add_module(MockModule('nand',['N1','N2','NO'],layer_size))
        self.add_module(MockMemoryModule(layer_size))
        for io_module in io_modules:
            self.add_module(io_module)
        self.add_module(MockGatingModule(mock_coding, self.get_layer_names()))
    def add_module(self, module):
        self.module_names.append(module.module_name)
        self.modules[module.module_name] = module
    def get_module(self, module_name):
        return self.modules[module_name]
    def get_layer_names(self):
        layer_names = []
        for name in self.modules:
            layer_names += self.modules[name].layer_names
        return layer_names
    def get_pattern(self, module_name, layer_name):
        return self.modules[module_name].get_pattern(layer_name)
    def list_patterns(self):
        pattern_list = []
        for name in self.module_names:
            pattern_list += [(name, layer_name, pattern) for (layer_name, pattern) in self.modules[name].list_patterns()]
        return pattern_list
    def hash_patterns(self):
        pattern_list = self.list_patterns()
        return {layer_name: pattern for (_, layer_name, pattern) in pattern_list}
    def set_pattern(self,layer_name, pattern):
        for module_name in self.modules:
            if layer_name in self.modules[module_name].layer_names:
                self.modules[module_name].set_pattern(layer_name, pattern)
                return
    def set_patterns(self, pattern_list):
        for (module_name, layer_name, pattern) in pattern_list:
            self.modules[module_name].set_pattern(layer_name, pattern)
    def tick(self):
        old_pattern_list = self.list_patterns()
        old_pattern_hash = self.hash_patterns()
        new_pattern_list = []
        for (module_name, layer_name, pattern) in old_pattern_list:
            if module_name in ['compare','nand','memory']:
                new_pattern = np.tanh(np.random.randn(len(pattern)))
                new_pattern_list.append((module_name, layer_name, new_pattern))
        self.set_patterns(new_pattern_list)
        # layer copies
        for to_layer_name in self.modules['gating'].net_layer_names:
            for from_layer_name in self.modules['gating'].net_layer_names:
                if self.modules['gating'].get_gate('V',to_layer_name, from_layer_name) > 0.5:
                    self.set_pattern(to_layer_name, old_pattern_hash[from_layer_name])
        self.modules['gating'].tick(old_pattern_hash)
        
class MockModule:
    def __init__(self, module_name, layer_names, layer_size=16):
        self.module_name = module_name
        self.layer_names = layer_names
        self.layer_size = layer_size
        self.layers = {name: -np.ones((layer_size,)) for name in layer_names}
        self.transitions = []
    def get_pattern(self, layer_name):
        return self.layers[layer_name].copy()
    def set_pattern(self, layer_name, layer_pattern):
        self.layers[layer_name] = layer_pattern.copy()
    def list_patterns(self):
        return [(layer_name, self.get_pattern(layer_name)) for layer_name in self.layer_names]
    def tick(self, pattern_hash):
        # module dynamics
        for pattern_list, next_pattern_list in self.transitions:
            matches = True
            for (_, layer_name, pattern) in pattern_list:
                if layer_name in pattern_hash and not eq(pattern_hash[layer_name], pattern):
                    matches = False
                    break
            if matches:
                for (module_name, layer_name, pattern) in next_pattern_list:
                    if module_name == self.module_name:
                        self.set_pattern(layer_name, pattern)
                break
    def learn(self, pattern_list, next_pattern_list):
        """
        train the module dynamics on the given transition.
        both pattern_lists should include patterns for every layer in this module.
        after training, when presented with the same patterns in pattern_list,
        the module layers will transition to the same patterns in next_pattern_list.
        """        
        pattern_list = [(module_name, layer_name, pattern.copy()) for (module_name, layer_name, pattern) in pattern_list]
        next_pattern_list = [(module_name, layer_name, pattern.copy()) for (module_name, layer_name, pattern) in next_pattern_list]
        self.transitions.insert(0,(pattern_list, next_pattern_list))

if __name__ == '__main__':
    mod = MockModule(module_name='mod', layer_names=['V'], layer_size=4)
    ones = np.ones(mod.layer_size)
    transitions = [
        [[('mod','V', -ones),('oth','U',+ones)],[('mod','V', +ones)]],
        [[('mod','V', +ones),('oth','U',+ones)],[('mod','V', -ones)]],
        [[('oth','U',-ones)],[('mod','V', -ones)]],
    ]
    for pattern_list, next_pattern_list in transitions:
        mod.learn(pattern_list, next_pattern_list)
    print('flashed')
    mod.set_pattern('V',-ones)
    pattern_hash = {'V':-ones, 'U': +ones}
    mod.tick(pattern_hash)
    print(mod.get_pattern('V'))
    pattern_hash = {'V':+ones, 'U': +ones}
    mod.tick(pattern_hash)
    print(mod.get_pattern('V'))
    pattern_hash = {'V':-ones, 'U': +ones}
    mod.tick(pattern_hash)
    print(mod.get_pattern('V'))
    pattern_hash = {'V':+ones, 'U': -ones}
    mod.tick(pattern_hash)
    print(mod.get_pattern('V'))
    pattern_hash = {'V':-ones, 'U': np.array([1,-1,1,-1])}
    mod.tick(pattern_hash)
    print(mod.get_pattern('V'))

class MockGatingModule(MockModule):
    def __init__(self, mock_coding, net_layer_names):
        MockModule.__init__(self, module_name='gating', layer_names=['V','L'], layer_size=len(net_layer_names)**2)
        self.coding = mock_coding
        self.net_layer_names = net_layer_names
        self.gate_index = {}
        index = 0
        for to_layer_name in self.net_layer_names:
            for from_layer_name in self.net_layer_names:
                self.gate_index[to_layer_name, from_layer_name] = index
                index += 1
    def set_gate(self, gate_layer_name, to_layer_name, from_layer_name, value):
        pattern = self.get_pattern(gate_layer_name)
        pattern[self.gate_index[to_layer_name,from_layer_name]] = value
        self.set_pattern(gate_layer_name, pattern)
    def get_gate(self, gate_layer_name, to_layer_name, from_layer_name):
        index = self.gate_index[to_layer_name,from_layer_name]
        return self.layers[gate_layer_name][index]
    # def tick(self, pattern_hash):
    #     self.set_pattern('V',-np.ones(self.layer_size)) # clear gates
    #     self.set_pattern('L',-np.ones(self.layer_size)) # clear gates
    #     opcode = pattern_hash['OPC']
    #     if eq(opcode, self.coding.encode('set')):
    #         to_layer_name = self.coding.decode(pattern_hash['OP2'])
    #         from_layer_name = 'OP1'
    #         self.set_gate('V',to_layer_name, from_layer_name, 0.9)

class MockMemoryModule(MockModule):
    def __init__(self, layer_size=16):
        MockModule.__init__(self, module_name='memory', layer_names=['K','V'], layer_size=layer_size)
    # def tick(self, layers, gates):
    #     # activity gates: key <- key, value <- key, value <- value, key <- copy, value <- copy
    #     # learning gates: key <- key, value <- key, value <- value
    #     pass

class MockIOModule(MockModule):
    def __init__(self, module_name, layer_size=16):
        MockModule.__init__(self, module_name, layer_names=['STDI','STDO'], layer_size=layer_size)
