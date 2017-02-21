import multiprocessing as mp
import numpy as np
import visualizer as vz
import mock_net as mn

class NVM:
    def __init__(self, coding, network):
        self.coding = coding
        self.network = network
        self.visualizing = False
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
        idxs = [1,2,4]
        self.viz_pipe.send(len(idxs))
        for i in idxs:
            self.viz_pipe.send('lay%d'%i) # name
            self.viz_pipe.send('asd%d'%(10+i)) # value
            pattern = np.random.randint(0,255,(32,),dtype=np.uint8).tobytes()
            self.viz_pipe.send_bytes(pattern) # bytes
    def show(self):
        self.hide() # flush any outdated viz process
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

def mock_nvm(layer_size=32):
    return NVM(mn.MockCoding(layer_size), mn.MockNet())

def run_viz(nvm_pipe):
    viz = vz.Visualizer(nvm_pipe)
    viz.launch()

if __name__ == '__main__':
    nvm = NVM()
    nvm.show()
    nvm.hide()

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
