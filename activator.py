import numpy as np

class Activator:
    def __init__(self, f, g, e, make_pattern, hash_pattern, on, off):
        self.f = f
        self.g = g
        self.e = e
        self.make_pattern = make_pattern
        self.hash_pattern = hash_pattern
        self.on = on
        self.off = off

def tanh_activator(pad, layer_size):
    return Activator(
        f = np.tanh,
        g = np.arctanh,
        e = lambda a, b: ((a > 0) == (b > 0)),
        make_pattern = lambda : (1.-pad)*np.sign(np.random.randn(layer_size,1)),
        hash_pattern = lambda p: (p > 0).tobytes(),
        on = 1. - pad,
        off = -(1. - pad))

def logistic_activator(pad, layer_size):
    return Activator(
        f = lambda v: .5*(np.tanh(v)+1),
        g = lambda v: np.arctanh(2*v-1),
        e = lambda a, b: ((a > .5) == (b > .5)),
        make_pattern = lambda : .5*((1.-pad)*np.sign(np.random.randn(layer_size,1)) + 1.),
        hash_pattern = lambda p: (p > .5).tobytes(),
        on = 1. - pad,
        off = 0. + pad)

def heaviside_activator(layer_size):
    return Activator(
        f = lambda v: (v > .5).astype(float),
        g = lambda v: (-1.)**(v < .5),
        e = lambda a, b: ((a > .5) == (b > .5)),
        make_pattern = lambda : (np.random.randn(layer_size,1) > 0).astype(float),
        hash_pattern = lambda p: (p > .5).tobytes(),
        on = 1.,
        off = 0.)
