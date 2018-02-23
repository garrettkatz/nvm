import numpy as np
from coder import Coder

class Sequencer:

    def __init__(self, layer_name, coder):
        self.layer_name = layer_name
        self.coder = coder
        self.transits = []

    def add_transit(self, new_state=None, old_state=None, **input_states):

        # Generate states if not provided, encode as necessary
        if new_state is None: new_state = self.coder.make_pattern()
        if type(new_state) is str: new_state = self.coder.encode(new_state)

        if old_state is None: old_state = np.zeros(new_state.shape)
        if type(old_state) is str: old_state = self.coder.encode(old_state)

        for layer, pattern in input_states.items():
            if type(pattern) is str:
                input_states[layer] = self.coder.encode(pattern)

        # Check for non-determinism
        for n, o, i in self.transits:
            # Different new state
            if (n != new_state).any(): continue
            # Different old state
            if (o != old_state).any(): continue
            # Different input layers
            if set(i.keys()) != set(input_states.keys()): continue
            # Different input patterns
            if any((i[l] != p).any() for l,p in input_states.items()): continue
            # Otherwise non-deterministic
            raise Exception("Created non-deterministic transit!")

        # Save transit
        self.transits.append((new_state, old_state, input_states))

        # Return new state
        return new_state
        
    def flash(self, f, g):
        
        # Unzip transits
        new_states, old_states, input_states = zip(*self.transits)
        P = len(self.transits)
        
        # Collect all input layer names and sizes
        layer_size = {self.layer_name: new_states[0].shape[0]}
        for states in input_states:
            for name, pattern in states.items():
                layer_size[name] = pattern.shape[0]
        
        # Fix layer order with sequence layer first
        names = layer_size.keys()
        names.remove(self.layer_name)
        names.insert(0, self.layer_name)

        # Populate transit matrices
        X = {self.layer_name : np.concatenate(old_states, axis=1)}
        for i, states in enumerate(input_states):
            for name, pattern in states.items():
                if name not in X: X[name] = np.zeros((pattern.shape[0], P))
                X[name][:, [i]] = pattern
        X = np.concatenate([X[name] for name in names], axis=0)
        Y = np.concatenate(new_states, axis=1)
        
        # Solve with hidden step
        W, bias, _ = zsolve(X, Y, f, g)
        
        # Split up weights by layer
        weights = {}
        offset = 0
        for name in names:
            weights[(self.layer_name, name)] = W[:,offset:offset + layer_size[name]]
            offset += layer_size[name]
        
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

    c = Coder(act.make_pattern, act.hash_pattern)
    
    s = Sequencer("gates", c)
    v_old = s.add_transit(new_state="SET")
    for to_layer in ["FEF","SC"]:
        for from_layer in ["FEF","SC"]:
            v_new = s.add_transit(
                new_state = to_layer + from_layer,
                old_state = v_old,
                op1 = to_layer, op2 = from_layer)

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
