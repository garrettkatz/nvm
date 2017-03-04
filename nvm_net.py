
class IOModule:
    def __init__(self, name, pipe, layer_size=32):
        self.name = name
        self.pipe = pipe
        self.input_layer = np.empty((layer_size,))
        self.output_layer = np.empty((layer_size,))
    def get_input_layer(self):
        # flush pipe and save last pattern
        while self.pipe.poll():
            pattern = self.pipe.recv_bytes()
            self.input_layer = np.fromstring(pattern)
        return self.input_layer.copy()
    def set_output_layer(self, pattern):
        self.output_layer = pattern.copy()
        self.pipe.send_bytes(pattern.tobytes())
    def get_layers(self):
        return [('input',self.get_input_layer()), ('output',self.output_layer.copy())]
    def set_layers(self, layer_patterns):
        pass
