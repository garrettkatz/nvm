class Layer:
    def __init__(self, name, shape, activator, coder):
        self.name = name
        self.shape = shape
        self.activator = activator
        self.coder = coder
        self.size = reduce(lambda x,y: x*y, shape)
