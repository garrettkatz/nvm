import numpy as np
from layer import Layer
from coder import Coder
from sequencer import Sequencer
from associator import Associator
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

        # Default to all off except internal gate activity
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
            gate_key = (self.gate_output.name, from_name, 'U')
            gate_value = self.gate_map.get_gate_value(gate_key, old_gates)
            if gate_value == self.gate_output.activator.off:
                raise Exception("Using input from layer that is not ungated!")

        # Include old hidden pattern in input for new hidden
        hidden_input_states = dict(input_states)
        hidden_input_states[self.gate_hidden.name] = old_hidden
        super(GateSequencer, self).add_transit(
            new_state=new_hidden, **hidden_input_states)

        # Ungate specified gates
        new_gates = self.make_gate_output(ungate)

        # add output to transit
        self.transit_outputs.append(new_gates)

        return new_hidden

    def stabilize(self, hidden, num_iters=1):
        """Stabilize activity for a few iterations"""
        for i in range(num_iters):
            hidden = self.add_transit(old_hidden=hidden)
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
    return [("MEM","MEM","D"), ("MEM","MEMH","U") ("MEMH","MEMH","U")]

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

    NL = len(layers) + 2 # +2 for gate out/hidden
    NG = NL**2 + NL
    NH = N
    actg = heaviside_activator(NG)
    acth = logistic_activator(PAD,NH)
    gate_output = Layer("gates", NG, actg, Coder(actg))
    gate_hidden = Layer("ghide", NH, acth, Coder(acth))
    layers.extend([gate_hidden, gate_output])
    
    gate_map = gm.make_nvm_gate_map(layers)
    gs = GateSequencer(gate_map, gate_output, gate_hidden,
        {layer.name: layer for layer in layers})

    go_start = gs.make_gate_output()
    gh_start = acth.make_pattern()
    gh = gs.stabilize(gh_start)
    # for reg in ["OPC","OP1","OP2","OP3"]:
    #     # load op from memory and step memory
    #     v = add_transit(X, Y, v, cpu_state(ungate = memu + cop(reg,"MEM")))
    #     v = stabilize(X, Y, v)

    gs.flash()
