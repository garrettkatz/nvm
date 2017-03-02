import multiprocessing as mp
import numpy as np
import visualizer as vz
import mock_net as mn

class IOLayer:
    """
    Figure out en/decode with human readable.
    hypervisor: one end of pipe, sends/recvs human-readable
    nvm: other end of pipe, decodes before send/encodes after receipt
    """
    def __init__(self, name, pipe, layer_size, coding=None):
        self.name = name
        self.pipe = pipe
        self.recv_layer = np.empty((layer_size,))
        self.send_layer = np.empty((layer_size,))
        self.coding = coding
    def recv_input_pattern():
        # flush pipe and save last pattern
        while self.pipe.poll():
            if self.coding is not None:
                token = self.pipe.recv()
                pattern = self.coding.encode(token)
            else:
                pattern = self.pipe.recv_bytes()
                pattern = np.fromstring(pattern)
            self.recv_layer = pattern.copy()
    def send_output_pattern(pattern):
        self.send_layer = pattern.copy()
        if self.coding is not None:
            self.pipe.send(self.coding.decode(pattern))            
        else:
            self.pipe.send_bytes(pattern.tobytes())
                
class NVM:
    def __init__(self, coding, network):
        self.coding = coding
        self.network = network
        self.visualizing = False
        # Encode layer names and constants
        symbols = self.network.layer_names + ['TRUE','FALSE','NIL','_']
        for symbol in symbols:
            self.coding.encode(symbol)
        # clear layers
        nil_pattern = self.coding.encode('_')
        layers = self.network.get_layers()
        layers = [(name, nil_pattern) for (name,_) in layers]
        self.network.set_layers(layers)
    def tick(self):
        # answer any visualizer request
        if self.visualizing:
            if self.viz_pipe.poll():
                # flush request
                self.viz_pipe.recv()
                # respond with data
                self.send_viz_data()
        # network update
        self.network.tick()
    def send_viz_data(self):
        """
        Protocol: <# layers>, <name>, <value>, <pattern>, <name>, <value>, <pattern>, ...
        """
        if not self.visualizing: return
        layers = self.network.get_layers()
        self.viz_pipe.send(len(layers))
        for (name, pattern) in layers:
            self.viz_pipe.send(name)
            self.viz_pipe.send(self.coding.decode(pattern)) # value
            pattern = (128*(pattern + 1.0)).astype(np.uint8).tobytes()
            self.viz_pipe.send_bytes(pattern) # bytes
    def show(self):
        self.hide() # flush any windowless viz process
        self.viz_pipe, other_end = mp.Pipe()
        self.viz_process = mp.Process(target=run_viz, args=(other_end,))
        self.viz_process.start()
        self.visualizing = True
        # send initial data for window layout
        self.send_viz_data()
    def hide(self):
        if not self.visualizing: return
        self.viz_pipe.send('shutdown')
        self.viz_process.join()
        self.viz_pipe = None
        self.viz_process = None
        self.visualizing = False

def mock_nvm(num_registers=3, layer_size=32):
    return NVM(mn.MockCoding(layer_size), mn.MockNet(num_registers, layer_size))

def run_viz(nvm_pipe):
    viz = vz.Visualizer(nvm_pipe)
    viz.launch()

if __name__ == '__main__':
    nvm = mock_nvm()
    # nvm.show()
    # nvm.hide()

# class NVM:
#     """
#     """
#     def __init__(self, net, encoding):
#         self.net = net
#         # in which process should encoding/decoding happen???
#         self.encoding = encoding
#         self.viz_on = False
#         self.pipe_to_viz = None
#         self.viz_process = None
#     def __str__(self):
#         return str(self.net)
#     def encode(self, human_readable):
#         return self.encoding.encode(human_readable)
#     def decode(self, machine_readable):
#         return self.encoding.decode(machine_readable)
#     def assemble(self, assembly_code):
#         pass
#     def load(self, object_code, label_table):
#         pass
#     def show_viz(self):
#         if not self.viz_on:
#             self.viz_on = True
#             self.pipe_to_viz, pipe_for_viz = mp.Pipe()
#             self.viz_process = mp.Process(target=run_viz, args=(pipe_for_viz,))
#             self.viz_process.start()
#     def hide_viz(self):
#         if self.viz_on:
#             self.pipe_to_viz.send('q')
#             self.viz_process.join()
#             self.viz_on = False
#     def viz_state(self):
#         activations = self.net.get_activations()
#         state = []
#         for layer_name in activations:
#             bits = tuple(np.uint8(255*(activations[layer_name]+1)/2))
#             state.append(tuple([layer_name, self.decode(activations[layer_name]), bits]))
#         return tuple(state)        
#     def tick(self):
#         # handle visualization
#         if self.viz_on:
#             if self.pipe_to_viz.poll():
#                 _ = self.pipe_to_viz.recv()
#                 self.pipe_to_viz.send(self.viz_state())
#         # execute instruction
#         self.net.tick()

# def run_viz(pipe_to_nvm):
#     nv.NVMViz(pipe_to_nvm, history=512)

# if __name__ == '__main__':
#     num_registers = 4
#     layer_size = 32
#     encoding = mn.MockEncoding(layer_size)
#     net = mn.MockNet(encoding, num_registers, layer_size, input_layers={})
#     nvm = NVM(net, encoding)
#     print(nvm)
#     # print(nvm.viz_state())
#     # nvm.show_viz()
#  # 
