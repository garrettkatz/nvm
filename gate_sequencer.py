import numpy as np
from layer import Layer
from coder import Coder
from sequencer import Sequencer
import gate_map as gm

PAD = 0.9
LAMBDA = np.arctanh(PAD)/PAD

class GateSequencer:
    def __init__(self, gate_map, gate_output, gate_hidden, input_layers):
        self.gate_map = gate_map
        self.gate_output = gate_output
        self.gate_hidden = gate_hidden
        self.output_sequencer = Sequencer(gate_output, input_layers)
        self.hidden_sequencer = Sequencer(gate_hidden, input_layers)
        
    def default_gate_output(self):
        """Default to all gates off except gate dynamics themselves"""
        pattern = self.gate_output.activator.off * np.ones((self.gate_output.size,1))
        gate_keys = [
            (self.gate_hidden.name, self.gate_hidden.name, 'U')
            (self.gate_output.name, self.gate_hidden.name, 'U')
            (self.gate_output.name, self.gate_output.name, 'D')]
        for k in gate_keys:
            pattern[self.gate_map.get_gate_index(k),0] = self.gate_output.activator.on
        return pattern

    def add_transit(self, ungate=[], new_hidden=None, old_gates=None, old_hidden=None, **input_states):
        # Default to random hidden patterns
        if old_hidden is None: old_hidden = self.gate_hidden.activator.make_pattern()
        if new_hidden is None: new_hidden = self.gate_hidden.activator.make_pattern()

        # Default old gates if not provided
        if old_gates is None: old_gates = self.default_gate_output()

        # Error if inputs are provided whose layers are not ungated
        for from_name in input_states:
            gate_key = (self.gate_output.name, from_name, 'U')
            gate_value = self.gate_map.get_gate_value(gate_key, old_gates)
            if gate_value == self.gate_output.activator.off:
                raise Exception("Using input from layer that is not ungated!")

        # Include old hidden pattern in input for new hidden
        hidden_input_states = dict(input_states)
        hidden_input_states[gate_hidden.name] = old_hidden
        self.hidden_sequencer.add_transit(
            new_state=new_hidden, **hidden_input_states)

        # Ungate specified gates
        new_gates = self.default_gate_output()
        for ug in ungate:
            new_gates[self.gate_map.get_gate_index(ug),0] = self.gate_output.activator.on

        # add transit
        self.output_sequencer.add_transit(
            new_state=new_gates, **{self.gate_hidden.name: old_hidden})

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

if __name__ == '__main__':

    np.set_printoptions(linewidth=200, formatter = {'float': lambda x: '% .2f'%x})

    N = 8
    PAD = 0.9

    from activator import *
    # act = tanh_activator(PAD, N)
    act = logistic_activator(PAD, N)
    coder = Coder(act)

    layer_names = ["A","B","C"]
    layers = [Layer(name, N, act, coder) for name in layer_names]
    
    gate_map = gm.make_nvm_gate_map(layers)
    NG = gate_map.get_gate_count()
    actg = logistic_activator(PAD,NG)
    NH = N
    acth = logistic_activator(PAD,NH)
    
    gate_output = Layer("gates", NG, actg, Coder(actg))
    gate_hidden = Layer("ghide", NH, acth, Coder(acth))
    
    gs = GateSequencer(gate_map, gate_output, gate_hidden, layers)
