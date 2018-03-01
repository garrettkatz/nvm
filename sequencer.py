import numpy as np
from layer import Layer
from coder import Coder

class Sequencer:

    def __init__(self, sequence_layer, input_layers):
        self.sequence_layer = sequence_layer
        self.input_layers = input_layers        
        self.transits = []

    def add_transit(self, new_state=None, **input_states):

        # Generate states if not provided, encode as necessary
        if new_state is None:
            new_state = self.sequence_layer.coder.make_pattern()
        if type(new_state) is str:
            new_state = self.sequence_layer.coder.encode(new_state)

        for name, pattern in input_states.items():
            if type(pattern) is str:
                input_states[name] = self.input_layers[name].coder.encode(pattern)

        # Check for non-determinism
        for n, i in self.transits:
            # Different new state
            if (n != new_state).any(): continue
            # Different input layers
            if set(i.keys()) != set(input_states.keys()): continue
            # Different input patterns
            if any((i[l] != p).any() for l,p in input_states.items()): continue
            # Otherwise non-deterministic
            raise Exception("Created non-deterministic transit!")

        # Save transit
        self.transits.append((new_state, input_states))

        # Return new state
        return new_state
        
    def flash(self, f, g):

        # Unzip transits
        all_new_states, all_input_states = zip(*self.transits)
        P = len(self.transits)

        # Populate input matrices
        X = {}
        for i, input_states in enumerate(all_input_states):
            for name, pattern in input_states.items():
                if name not in X: X[name] = np.zeros((pattern.shape[0], P))
                X[name][:, [i]] = pattern

        # Fixed layer order
        names = X.keys()
        
        # Solve with hidden step
        X = np.concatenate([X[name] for name in names], axis=0)
        Y = np.concatenate(all_new_states, axis=1)
        W, bias, _ = zsolve(X, Y, f, g)
        
        # Split up weights by layer
        weights = {}
        offset = 0
        for name in names:
            layer_size = self.input_layers[name].size
            weights[(self.sequence_layer.name, name)] = W[:,offset:offset + layer_size]
            offset += layer_size
        
        # return final weights and bias
        return weights, bias

def zsolve(X, Y, f, g, verbose=True):
    """
    Construct W that transitions states in X to corresponding states in Y
    X, Y are arrays, with paired activity patterns as columns
    f, g are the activation function and its inverse    
    To deal with low-rank X, each transition uses an intermediate "hidden step"
    """

    # size of layer being sequenced
    N = Y.shape[0]
    
    # for low-rank X, get coefficients A of X's column space
    _, sv, A = np.linalg.svd(X, full_matrices=False)
    rank_tol = sv.max() * max(X.shape) * np.finfo(sv.dtype).eps # from numpy
    A = A[sv > rank_tol, :]
    
    # use A to set intermediate Z that is low-rank pre non-linearity
    Z = np.zeros(X.shape)
    Z[:N,:] = f(np.random.randn(N, A.shape[0]).dot(A))

    # solve linear equations
    XZ = np.concatenate((
        np.concatenate((X, Z), axis=1),
        np.ones((1,2*X.shape[1])) # bias
    ), axis = 0)
    ZY = np.concatenate((Z[:N,:], Y), axis=1)
    W = np.linalg.lstsq(XZ.T, g(ZY).T, rcond=None)[0].T

    # weights, bias, hidden
    return W[:,:-1], W[:,[-1]], Z

if __name__ == '__main__':

    np.set_printoptions(linewidth=200, formatter = {'float': lambda x: '% .2f'%x})

    N = 8
    PAD = 0.9

    from activator import *
    # act = tanh_activator(PAD, N)
    act = logistic_activator(PAD, N)

    c = Coder(act)
    
    g = Layer("gates",N, act, c)
    input_layers = {name: Layer(name, N, act, c) for name in ["gates","op1","op2"]}
    s = Sequencer(g, input_layers)
    v_old = s.add_transit(new_state="SET")
    for to_layer in ["FEF","SC"]:
        for from_layer in ["FEF","SC"]:
            v_new = s.add_transit(
                new_state = to_layer + from_layer,
                gates = v_old, op1 = to_layer, op2 = from_layer)

    print(c.list_tokens())

    weights, bias = s.flash(act.f, act.g)
    for k,w in weights.items():
        print(k)
        print(w)
    print("bias")
    print(bias.T)

    a = {"gates":v_old, "op1":c.encode("SC"), "op2":c.encode("SC")}
    wv = bias.copy()
    for (_,from_layer),w in weights.items(): wv += w.dot(a[from_layer])
    z = np.zeros(v_old.shape)
    a = {"gates":act.f(wv), "op1": z, "op2":z}
    wv = bias.copy()
    for (_,from_layer),w in weights.items(): wv += w.dot(a[from_layer])
    v_test = act.f(wv)

    for v in [v_old, v_test, v_new]:
        print(c.decode(v), v.T)
    print(act.e(v_test, v_new).T)
