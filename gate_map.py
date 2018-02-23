class GateMap:
    def __init__(self, layer_names):
    
        # map between keys and indices
        gate_keys = [] # indices to keys
        gate_index = {} # keys to indices
        
        # set up map
        for to_layer in layer_names:
        
            # Add within-layer decay gate
            gate_index[(to_layer, to_layer, "D")] = len(gate_keys)
            gate_keys.append((to_layer, to_layer, "D"))

            # Add inter-layer update gates
            for from_layer in layer_names:
                gate_index[(to_layer, from_layer, "U")] = len(gate_keys)
                gate_keys.append((to_layer, from_layer, "U"))

        # save map
        self.gate_keys, self.gate_index = gate_keys, gate_index

    def get_gate_count(self):
        return len(self.gate_index)

    def get_gate_index(self, to_layer, from_layer, gate_type):
        """
        Returns the gate index for a given layer pair and type (U or D)
        """
        return self.gate_index[to_layer, from_layer, gate_type]
        
    def get_gate_value(self, gate_key, gate_pattern):
        """
        Returns the activation value in gate_pattern for a particular gate_key
        """
        return p[self.gate_index[gate_key], 0]
