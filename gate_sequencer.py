import numpy as np
from layer import Layer
from coder import Coder
from sequencer import Sequencer
import gate_map as gm

PAD = 0.9
LAMBDA = np.arctanh(PAD)/PAD

class GateSequencer(Sequencer, object):
    def __init__(self, gate_map, gate_output, gate_hidden, input_layers):
        self.gate_map = gate_map
        self.gate_output = gate_output
        super(GateSequencer, self).__init__(gate_hidden, input_layers)
        self.gate_hidden = self.sequence_layer
        self.transit_outputs = []

    def make_gate_output(self, ungate=[]):
        """Make gate output pattern where specified gate key units are on"""

        # Internal gate activity always on
        pattern = self.gate_output.activator.off * np.ones((self.gate_output.size,1))
        gate_keys = [
            (self.gate_hidden.name, self.gate_hidden.name, 'U'),
            (self.gate_output.name, self.gate_hidden.name, 'U'),
            (self.gate_output.name, self.gate_output.name, 'D')]

        # Ungate provided keys
        for k in gate_keys + ungate:
            i = self.gate_map.get_gate_index(k)
            pattern[i,0] = self.gate_output.activator.on

        return pattern

    def add_transit(self, ungate=[], new_hidden=None, old_gates=None, old_hidden=None, **input_states):

        # Default to random hidden patterns
        if old_hidden is None: old_hidden = self.gate_hidden.activator.make_pattern()
        if new_hidden is None: new_hidden = self.gate_hidden.activator.make_pattern()

        # Default old gates if not provided
        if old_gates is None: old_gates = self.make_gate_output()

        # Error if inputs are provided whose layers are not ungated
        for from_name in input_states:
            gate_key = (self.gate_hidden.name, from_name, 'U')
            gate_value = self.gate_map.get_gate_value(gate_key, old_gates)
            if gate_value == self.gate_output.activator.off:
                raise Exception(
                    "Using input from layer that is not ungated! Expected "+str(gate_key))

        # Error if ungated layers not provided as input
        for p in range(old_gates.shape[0]):
            if old_gates[p,0] == self.gate_output.activator.on:
                gate_key = self.gate_map.get_gate_key(p)
                to_name, from_name, gate_type = gate_key
                if not to_name == self.gate_hidden.name: continue
                if not gate_type == 'U': continue
                if from_name in input_states: continue
                if from_name == self.gate_hidden.name: continue
                raise Exception("No input provided for ungated layer!  Expected "+str(gate_key))

        # Include old hidden pattern in input for new hidden
        hidden_input_states = dict(input_states)
        hidden_input_states[self.gate_hidden.name] = old_hidden
        super(GateSequencer, self).add_transit(
            new_state=new_hidden, **hidden_input_states)

        # Ungate specified gates
        new_gates = self.make_gate_output(ungate)
        self.transit_outputs.append(new_gates)

        return new_gates, new_hidden

    def stabilize(self, hidden, num_iters=1):
        """Stabilize activity for a few iterations"""
        for i in range(num_iters):
            _, hidden = self.add_transit(old_hidden=hidden)
        return hidden

    def flash(self):
        
        # Flash sequencer
        weights, bias, XYZ = super(GateSequencer, self).flash()

        # Form output associations
        X, _, Z = XYZ
        X = X[:self.gate_hidden.size,:] # hidden layer portions
        Z = Z[:self.gate_hidden.size,:] # hidden layer portions
        Y = np.concatenate((
            self.make_gate_output()*np.ones((1, X.shape[1])), # intermediate output
            np.concatenate(self.transit_outputs, axis=1), # transit output
        ),axis=1)

        # solve linear equations for output
        XZ = np.concatenate((
            np.concatenate((X, Z), axis=1),
            np.ones((1,2*X.shape[1])) # bias
        ), axis = 0)
        g = self.gate_output.activator.g
        W = np.linalg.lstsq(XZ.T, g(Y).T, rcond=None)[0].T

        # update weights and bias with output
        weights[(self.gate_output.name, self.gate_hidden.name)] = W[:,:-1]
        bias[self.gate_output.name] = W[:,[-1]]
    
        # weights, bias, hidden
        return weights, bias

def gcopy(to_layer, from_layer):
    """gates for inter-layer signal"""
    return [(to_layer,from_layer,"U"), (to_layer,to_layer,"D")]

def gmem():
    """gates for memory (token and hidden) layer updates"""
    return [("mem","mem","D"), ("mem","memh","U"), ("memh","memh","U")]

if __name__ == '__main__':

    np.set_printoptions(linewidth=200, formatter = {'float': lambda x: '% .2f'%x})

    N = 8
    PAD = 0.9

    from activator import *
    # act = tanh_activator(PAD, N)
    act = logistic_activator(PAD, N)
    coder = Coder(act)

    layer_names = ["mem","memh","opc","op1","op2","op3"]
    layers = [Layer(name, N, act, coder) for name in layer_names]

    NL = len(layers) + 2 # +2 for gate out/hidden
    NG = NL**2 + NL
    NH = 100
    actg = heaviside_activator(NG)
    acth = logistic_activator(PAD,NH)
    gate_output = Layer("gates", NG, actg, Coder(actg))
    gate_hidden = Layer("ghide", NH, acth, Coder(acth))
    layers.extend([gate_hidden, gate_output])
    
    gate_map = gm.make_nvm_gate_map(layers)
    gs = GateSequencer(gate_map, gate_output, gate_hidden,
        {layer.name: layer for layer in layers})

    h_start = acth.make_pattern()
    h = h_start
    for reg in ["opc"]:#,"op1","op2","op3"]:
        # load op from memory and step memory
        _, h = gs.add_transit(ungate = gmem() + gcopy(reg,"mem"), old_hidden = h)
        h = gs.stabilize(h)

    # Let opcode bias the gate layer
    g, h = gs.add_transit(ungate = [("ghide","opc","U")], old_hidden=h)
    
    # Ready to execute instruction
    h_ready = h.copy()
    
    # _, _ = gs.add_transit(
    #     new_hidden = h_start, # begin next clock cycle
    #     old_gates = g, old_hidden = h_ready)

    
    ###### NOP instruction
    
    g, h = gs.add_transit(
        new_hidden = h_start, # begin next clock cycle
        old_gates = g, old_hidden = h_ready, opc = "nop")

    weights, bias = gs.flash()

    h = h_start
    for i in range(30):
        h = acth.f(weights[("ghide","ghide")].dot(h) + bias["ghide"])
        print(i, acth.e(h, h_ready).all())
