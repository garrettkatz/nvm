import multiprocessing as mp
import Tkinter as tk
import PIL as pil
import PIL.ImageTk as itk # on fedora 24 needed $ dnf install python-pillow-tk
import numpy as np
import time

class Visualizer:
    def __init__(self, nvm_pipe):
        self.nvm_pipe = nvm_pipe
        self.nvm_shutdown = False
    def launch(self):
        # Set up window
        self.tk_root = tk.Tk()
        self.tk_frame = tk.Frame(self.tk_root, width=100, height=100)
        self.tk_frame.pack()
        # Launch main tk loop
        self.tk_root.after(0,self.update)
        self.tk_root.mainloop()
        # Continue process until shutdown by nvm
        while not self.nvm_shutdown:
            message = self.recv()
            if message == 'shutdown':
                self.nvm_shutdown = True
    def recv(self):
        """
        Protocol: <# layers>, <name>, <value>, <pattern>, <name>, <value>, <pattern>, ...
        """
        message = self.nvm_pipe.recv()
        if message != 'shutdown':
            num_layers = message
            message = []
            for _ in range(num_layers):
                name = self.nvm_pipe.recv()
                value = self.nvm_pipe.recv()
                pattern = self.nvm_pipe.recv_bytes()
                pattern = np.fromstring(pattern,dtype=np.uint8)
                message.append((name, value, pattern))
        return message
    def update(self):
        if self.nvm_pipe.poll():
            message = self.recv()
            if message == 'shutdown':
                self.nvm_shutdown = True
                self.tk_root.destroy()
                return
            self.tk_root.update()
        self.tk_root.after(5, self.update)
