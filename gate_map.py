class GateMap:
    def __init__(self, gate_keys):
    
        # map between keys and indices
        self.gate_keys = list(gate_keys) # indices to keys
        self.gate_index = {k:i
            for i,k in enumerate(gate_keys)} # keys to indices

    def get_gate_count(self):
        return len(self.gate_index)

    def get_gate_index(self, gate_key):
        """Returns the gate index for a particular key"""
        return self.gate_index[gate_key]

    def get_gate_key(self, gate_index):
        """Returns the gate key for a particular index"""
        return self.gate_keys[gate_index]
        
    def get_gate_value(self, gate_key, gate_pattern):
        """
        Returns the activation value in gate_pattern for a particular gate_key
        """
        return gate_pattern[self.gate_index[gate_key], 0]
    
def make_nvm_gate_map(layers):

    # set up gate keys
    gate_keys = []
    for to_layer in layers:
    
        # Add within-layer decay gate
        gate_keys.append((to_layer.name, to_layer.name, "D"))

        # Add inter-layer update gates
        for from_layer in layers:
            gate_keys.append((to_layer.name, from_layer.name, "U"))

    return GateMap(gate_keys)
