import numpy as np
from layer import Layer
from coder import Coder

class Associator:

    def __init__(self, output_layer, input_layers):
        self.output_layer = output_layer
        self.input_layers = input_layers
        self.associations = []

    def add_association(self, output_state=None, **input_states):

        # Generate states if not provided, encode as necessary
        if output_state is None:
            output_state = self.output_layer.activator.make_pattern()
        if type(output_state) is str:
            output_state = self.output_layer.coder.encode(output_state)

        for name, pattern in input_states.items():
            if type(pattern) is str:
                input_states[name] = self.input_layers[name].coder.encode(pattern)

        # Check for non-determinism
        for o, i in self.associations:
            # Different output state
            if (o != output_state).any(): continue
            # Different input layers
            if set(i.keys()) != set(input_states.keys()): continue
            # Different input layer patterns
            if any((i[l] != p).any() for l,p in input_states.items()): continue
            # Otherwise non-deterministic
            raise Exception("Created non-deterministic association!")

        # Save association
        self.associations.append((output_state, input_states))

        # Return new state
        return output_state
        
    def flash(self):

        # Unzip associations
        all_output_states, all_input_states = zip(*self.associations)
        P = len(self.associations)

        # Populate input matrices
        X = {}
        for i, input_states in enumerate(all_input_states):
            for name, pattern in input_states.items():
                if name not in X: X[name] = np.zeros((pattern.shape[0], P))
                X[name][:, [i]] = pattern

        # Fixed layer order
        names = X.keys()
        
        # Solve with one step
        X = np.concatenate([X[name] for name in names], axis=0)
        Y = np.concatenate(all_output_states, axis=1)
        W, bias = osolve(X, Y,
            self.output_layer.activator.f,
            self.output_layer.activator.g)
        
        # Split up weights by layer
        weights = {}
        offset = 0
        for name in names:
            layer_size = self.input_layers[name].size
            weights[(self.output_layer.name, name)] = W[:,offset:offset + layer_size]
            offset += layer_size
        
        # return final weights and bias
        return weights, bias

def osolve(X, Y, f, g, verbose=True):
    """
    Construct W that associates states in X to corresponding states in Y
    X, Y are arrays, with paired activity patterns as columns
    f, g are the activation function and its inverse    
    Error for X that are lower rank than Y
    """

    # size of layer being sequenced
    N = Y.shape[0]
    
    # check matrix ranks
    rank_X = np.linalg.matrix_rank(X)
    rank_Y = np.linalg.matrix_rank(Y)
    if rank_Y > rank_X:
        raise Exception('Associated rank %d X with rank %d Y!'%(rank_X, rank_Y))

    # solve linear equations with bias
    Xb = np.concatenate((X,
        np.ones((1,X.shape[1])) # bias
    ), axis = 0)
    W = np.linalg.lstsq(Xb.T, g(Y).T, rcond=None)[0].T

    # weights, bias
    return W[:,:-1], W[:,[-1]]

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
    assoc = Associator(g, input_layers)
    v_old = g.coder.encode("SET")
    for to_layer in ["FEF","SC"]:
        for from_layer in ["FEF","SC"]:
            v_new = assoc.add_association(
                output_state = to_layer + from_layer,
                gates = v_old, op1 = to_layer, op2 = from_layer)
        weights, bias = assoc.flash()

        print(c.list_tokens())
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
