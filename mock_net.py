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
        register_names = ['IP','OPC','OP1','OP2','OP3'] + ['{%d}'%r for r in range(num_registers)]
        self.control_module = MockModule(register_names, layer_size)
        self.compare_module = MockModule(['C1','C2','CO'],layer_size)
        self.nand_module = MockModule(['N1','N2','NO'],layer_size)
        self.memory_module = MockModule(['K','V'],layer_size)
        self.io_modules = io_modules
    def get_layers(self):
        layers = []
        layers += self.control_module.get_layers()
        layers += self.compare_module.get_layers()
        layers += self.nand_module.get_layers()
        layers += self.memory_module.get_layers()
        for io_module in self.io_modules:
            layers += io_module.get_layers()
        return layers
    def set_layers(self, layers):
        for (name, pattern) in layers.:
            self.layers[name] = pattern.copy()
    def tick(self):
        self.layers = {name: np.tanh(np.random.randn(self.layer_size)) for name in self.layer_names}
        pass
        
class MockModule:
    def __init__(self, layer_names, layer_size=32):
        self.layer_names = layer_names
        self.layer_size = layer_size
        self.layers = {name: -np.ones((layer_size,)) for name in layer_names}
    def get_layers(self):
        return [(name, self.layers[name].copy()) for name in self.layer_names]
    def set_layers(self, layer_patterns):
        for (name, pattern) in layer_patterns:
            self.layers[name] = pattern.copy()
    

# class MockNet:
#     def __init__(self, encoding, num_registers, layer_size=32, input_layers={}, output_layers={}):
#         self.encoding = encoding
#         self.num_registers = num_registers
#         self.layer_size = layer_size
#         self.layer_groups = [['IP','OPC','OP1','OP2','OP3'],
#                             ['{%d}'%r for r in range(num_registers)],
#                             ['K','V'],
#                             ['C1','C2','CO'],
#                             ['N1','N2','NO'],
#                             input_layers.keys(),
#                             output_layers.keys()]
#         self.layers = {}
#         for layer_group in self.layer_groups:
#             for layer_name in layer_group:
#                 self.layers[layer_name] = np.empty((layer_size,))
#     def encode(self, human_readable):
#         return self.encoding.encode(human_readable)
#     def decode(self, machine_readable):
#         return self.encoding.decode(machine_readable)
#     def get_layers(self):
#         return {layer_name: activity.copy() for (layer_name, activity) in self.layers.iteritems()}
#     def __str__(self):
#         text = ''
#         for group in self.layer_groups:
#             text += '[' + ', '.join(group) + ']: '
#             text += '[' + ', '.join(self.decode(self.layers[layer_name]) for layer_name in group) + ']\n'
#         return text
#     def tick(self):
#         pass
