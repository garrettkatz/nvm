import numpy as np

def bits(key, layer_size):
    machine_readable = np.empty((self.layer_size,))
    for unit in range(self.layer_size):
        machine_readable[unit] = (-1)**(1 & (key >> unit))
    return machine_readable

class MockEncoding:
    """
    Mapping between machine- and human-readable constants
    """
    def __init__(self, layer_size):
        self.layer_size = layer_size
        self.next_key = -1
        self.machine_readable_of = {}
        self.human_readable_of = {}
    def encode(self, human_readable):
        if human_readable in self.machine_readable_of:
            return self.machine_readable_of[human_readable]
        key = self.next_key
        self.next_key -= 1
        machine_readable = np.empty((self.layer_size,))
        for unit in range(self.layer_size):
            machine_readable[unit] = (-1)**(1 & (key >> unit))
        machine_readable = machine_readable
        self.machine_readable_of[human_readable] = machine_readable
        self.human_readable_of[tuple(machine_readable)] = human_readable
        return machine_readable
    def decode(self, machine_readable):
        if tuple(machine_readable) not in self.human_readable_of:
            return '<?>'
        return self.human_readable_of[tuple(machine_readable)]

class MockNet:
    def __init__(self, encoding, num_registers, layer_size=32, input_layers={}, output_layers={}):
        self.registers = {'{%d}'%r: np.empty((layer_size,)) for r in range(num_registers)}
    def encode(self, human_readable):
        return self.encoding.encode(human_readable)
    def decode(self, machine_readable):
        return self.encoding.decode(machine_readable)
    def get_activations(self):
        pass
    def __str__(self):
        activations = self.get_activations()
        layer_groups = [['IP','OPC','OP1','OP2','OP3'],
                        ['{%d}'%r for r in range(self.num_registers)],
                        ['K','V'],
                        ['C1','C2','CO'],
                        ['N1','N2','NO']]
        text = ''
        for group in layer_groups:
            text += '[' + ', '.join(group) + ']: '
            text += '[' + ', '.join(self.encoding.decode(activations[layer_name]) for layer_name in group) + ']\n'
        return text
    def tick(self):
        pass
