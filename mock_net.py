import numpy as np

class MockLayer:
    def __init__(self, name, size):
        self.name = name
        self.size = size
    def adapt(self, activity):
        pass
    def update(self, activity):
        return activity[self.name]

class MockNet:
    def __init__(self, layers):
        self.layers = layers
        self.activity = {}
        for ell in self.layers:
            layer = self.layers[ell]
            activity = np.empty((layer.size,))
            self.activity[ell] = activity
    def tick(self):
        new_activity = {}
        for ell in self.layers:
            self.layers[ell].adapt(self.activity)
            new_activity[ell] = self.layers[ell].activate(self.activity)
        self.activity = new_activity
