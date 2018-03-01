class Layer:
    def __init__(self, name, size, activator, coder):
        self.name = name
        self.size = size
        self.activator = activator
        self.coder = coder
