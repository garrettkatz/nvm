import numpy as np

class GateSequencer:
    def __init__(self, gate_map, name_o, name_h, N_h, coder):
        self.gate_map = gate_map
        self.N_o = gate_map.get_gate_count()
        self.N_h = N_h
        self.seq_o = Sequencer(name_o, coder)
        self.seq_h = Sequencer(name_h, coder)
    
    def add_transit(self, ungate=[], old_hidden, **input_states):
        pass

def default_gates():
    """All closed except the (gate,gate) update"""
    gates = -PAD*np.ones((N_GH,1))
    gates[GATE_INDEX[("GATES","GATES","U")],0] = PAD
    return gates

def gcopy(to_layer, from_layer):
    """gates for inter-layer signal"""
    return [(to_layer,from_layer,"U"), (to_layer,to_layer,"D")]

def gmem():
    """gates for memory (token and hidden) layer updates"""
    return [("MEM","MEM","D"), ("MEM","MEMH","U") ("MEMH","MEMH","U")]

def stabilize(X, Y, v):
    """Stabilize memory for a few iterations"""
    for s in range(1):
        v = add_transit(X, Y, v, cpu_state())
    return v
