import numpy as np

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
        # return encoding if already encoded
        if human_readable in self.machine_readable_of:
            return self.machine_readable_of[human_readable]
        # encode with next key
        key = self.next_key
        self.next_key -= 1
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
    def __init__(self, encoding, num_registers, layer_size=32, input_layers={}, output_layers={}):
        self.encoding = encoding
        self.num_registers = num_registers
        self.layer_size = layer_size
        self.layer_groups = [['IP','OPC','OP1','OP2','OP3'],
                            ['{%d}'%r for r in range(num_registers)],
                            ['K','V'],
                            ['C1','C2','CO'],
                            ['N1','N2','NO'],
                            input_layers.keys(),
                            output_layers.keys()]
        self.layers = {}
        for layer_group in self.layer_groups:
            for layer_name in layer_group:
                self.layers[layer_name] = np.empty((layer_size,))
    def encode(self, human_readable):
        return self.encoding.encode(human_readable)
    def decode(self, machine_readable):
        return self.encoding.decode(machine_readable)
    def get_layers(self):
        return {layer_name: activity.copy() for (layer_name, activity) in self.layers.iteritems()}
    def __str__(self):
        text = ''
        for group in self.layer_groups:
            text += '[' + ', '.join(group) + ']: '
            text += '[' + ', '.join(self.decode(self.layers[layer_name]) for layer_name in group) + ']\n'
        return text
    def tick(self):
        pass
