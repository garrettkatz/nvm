import multiprocessing as mp
import matplotlib.pyplot as plt
import Tkinter as tk
import PIL as pil
import PIL.ImageTk as itk # on fedora 24 needed $ dnf install python-pillow-tk
import numpy as np
import time
import mock_net as mn
import nvm_viz as nv

class NVM:
    """
    """
    def __init__(self, net, encoding):
        self.net = net
        # in which process should encoding/decoding happen???
        self.encoding = encoding
        self.viz_on = False
        self.pipe_to_viz = None
        self.viz_process = None
    def __str__(self):
        return str(self.net)
    def encode(self, human_readable):
        return self.encoding.encode(human_readable)
    def decode(self, machine_readable):
        return self.encoding.decode(machine_readable)
    def assemble(self, assembly_code):
        pass
    def load(self, object_code, label_table):
        pass
    def show_viz(self):
        if not self.viz_on:
            self.viz_on = True
            self.pipe_to_viz, pipe_for_viz = mp.Pipe()
            self.viz_process = mp.Process(target=run_viz, args=(pipe_for_viz,))
            self.viz_process.start()
    def hide_viz(self):
        if self.viz_on:
            self.pipe_to_viz.send('q')
            self.viz_process.join()
            self.viz_on = False
    def viz_state(self):
        activations = self.net.get_activations()
        state = []
        for layer_name in activations:
            bits = tuple(np.uint8(255*(activations[layer_name]+1)/2))
            state.append(tuple([layer_name, self.decode(activations[layer_name]), bits]))
        return tuple(state)        
    def tick(self):
        # handle visualization
        if self.viz_on:
            if self.pipe_to_viz.poll():
                _ = self.pipe_to_viz.recv()
                self.pipe_to_viz.send(self.viz_state())
        # execute instruction
        self.net.tick()

def run_viz(pipe_to_nvm):
    nv.NVMViz(pipe_to_nvm, history=512)

if __name__ == '__main__':
    num_registers = 4
    layer_size = 32
    encoding = mn.MockEncoding(layer_size)
    net = mn.MockNet(encoding, num_registers, layer_size, input_layers={'in':mn.MockLayer('in',layer_size)})
    nvm = NVM(net, encoding)
    print(nvm)
    # print(nvm.viz_state())
    nvm.show_viz()
