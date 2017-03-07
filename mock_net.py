import numpy as np

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
    def __init__(self, num_registers, layer_size=32, io_modules=[]):
        self.num_registers = num_registers
        self.layer_size = layer_size
        self.modules = {}
        register_names = ['IP','OPC','OP1','OP2','OP3'] + ['{%d}'%r for r in range(num_registers)]
        self.modules['control'] = MockModule('control',register_names, layer_size)
        self.modules['compare'] = MockModule('compare',['C1','C2','CO'],layer_size)
        self.modules['nand'] = MockModule('nand',['N1','N2','NO'],layer_size)
        # self.modules['memory'] = MockModule('memory',['K','V'],layer_size)
        self.modules['memory'] = MockMemoryModule(layer_size)
        self.module_names = ['control','compare','nand','memory']
        for io_module in io_modules:
            self.modules[io_module.module_name] = io_module
            self.module_names.append(io_module.module_name)
    def get_layer_names(self):
        layer_names = []
        for name in self.modules:
            layer_names += self.modules[name].layer_names
        return layer_names
    def get_layer(self, module_name, layer_name):
        return self.modules[module_name].get_layer(layer_name)
    def get_layers(self):
        layers = []
        for name in self.module_names:
            layers += [(name, layer_name, pattern) for (layer_name, pattern) in self.modules[name].get_layers()]
        return layers
    def set_layers(self, layers):
        for (module_name, layer_name, pattern) in layers:
            self.modules[module_name].set_layer(layer_name, pattern)
    def tick(self):
        old_layers = self.get_layers()
        new_layers = []
        for (module_name, layer_name, pattern) in old_layers:
            if module_name in ['control','compare','nand','memory']:
                new_layers.append((module_name, layer_name, np.tanh(np.random.randn(self.layer_size))))
        new_layers.append(('stdio', 'in', self.get_layer('stdio','in')))
        self.set_layers(new_layers)
        
class MockModule:
    def __init__(self, module_name, layer_names, layer_size=32):
        self.module_name = module_name
        self.layer_names = layer_names
        self.layer_size = layer_size
        self.layers = {name: -np.ones((layer_size,)) for name in layer_names}
    def get_layer(self, layer_name):
        return self.layers[layer_name].copy()
    def set_layer(self, layer_name, layer_pattern):
        self.layers[layer_name] = layer_pattern.copy()
    def get_layers(self):
        return [(name, self.layers[name].copy()) for name in self.layer_names]
    def set_layers(self, layer_patterns):
        for (name, pattern) in layer_patterns:
            self.layers[name] = pattern.copy()

class MockMemoryModule(MockModule):
    def __init__(self, layer_size=32):
        MockModule.__init__(self, module_name='memory', layer_names=['key','value'], layer_size=32)
    def tick(self, layers, gates):
        # activity update
        # weight update
        pass

class MockIOModule(MockModule):
    pass
