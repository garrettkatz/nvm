import multiprocessing as mp
import numpy as np
import visualizer as vz
import mock_net as mn

class NVM:
    def __init__(self, coding, network):
        self.coding = coding
        self.network = network
        self.visualizing = False
        # Encode layer names and constants
        symbols = self.network.get_layer_names() + ['TRUE','FALSE','NIL','_']
        for symbol in symbols:
            self.coding.encode(symbol)
        # clear layers
        layers = self.network.get_layers()
        layers = [(module_name, layer_name, self.coding.encode('_')) for (module_name, layer_name,_) in layers]
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
        for (_, layer_name, pattern) in layers:
            self.viz_pipe.send(layer_name)
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
    def set_input(self, message, io_module_name, from_human_readable=True):
        if from_human_readable:
            pattern = self.coding.encode(message)
        else:
            pattern = np.fromstring(pattern,dtype=float)
        self.network.set_layers([(io_module_name, 'in', pattern)])
    def get_output(self, io_module_name, to_human_readable=True):
        pattern = self.network.get_layer(io_module_name, 'out')
        if to_human_readable:
            message = self.coding.decode(pattern)
        else:
            message = pattern.tobytes()
        return message

def mock_nvm(num_registers=3, layer_size=32):
    stdio = mn.MockModule('stdio', ['in','out'], layer_size)
    net = mn.MockNet(num_registers, layer_size, io_modules=[stdio])
    return NVM(mn.MockCoding(layer_size), net)

def run_viz(nvm_pipe):
    viz = vz.Visualizer(nvm_pipe)
    viz.launch()

if __name__ == '__main__':
    nvm = mock_nvm()
    nvm.set_input('blah','stdio',from_human_readable=True)
    print(nvm.get_output('stdio',to_human_readable=True))
    # nvm.show()
    # nvm.hide()
