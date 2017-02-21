import multiprocessing as mp
import Tkinter as tk
import PIL as pil
import PIL.ImageTk as itk # on fedora 24 needed $ dnf install python-pillow-tk
import numpy as np
import time

class Visualizer:
    def __init__(self, nvm_pipe, history=256, padding=8, label_padding=40):
        self.nvm_pipe = nvm_pipe
        self.nvm_shutdown = False
        self.history = history
        self.padding = padding
        self.label_padding = label_padding
    def launch(self):
        # Receive initial state for window layout
        message = self.recv()
        if message == 'shutdown':
            self.nvm_shutdown = True
            return
        # Set up window
        width = self.label_padding + self.padding + self.history + self.padding
        height = (1+len(message))*self.padding + sum([len(pattern) for (_,_,pattern) in message])
        self.tk_root = tk.Tk()
        self.tk_frame = tk.Frame(self.tk_root, width=width, height=height)
        # Set up raster layout
        self.rasters, self.canvases, self.images, self.photos, self.labels = [], [], [], [], []
        y = self.padding # raster position
        for (name, value, pattern) in message:
            width, height = self.history, len(pattern)
            raster = np.zeros((height, width), dtype=np.uint8)
            canvas = tk.Canvas(self.tk_frame, width=width, height=height)
            label = tk.Label(self.tk_frame, text='%s\n%s'%(name,value), justify=tk.LEFT)
            canvas.place(x=self.padding+self.label_padding,y=y)
            label.place(x=0,y=y)
            self.rasters.append(raster)
            self.canvases.append(canvas)
            self.labels.append(label)
            self.images.append(None)
            self.photos.append(None)
            y += height + self.padding
        self.tk_frame.pack()
        # Request more data and launch main tk loop
        self.request()
        self.tk_root.after(0,self.update)
        self.tk_root.mainloop()
        # Continue process until shutdown by nvm
        while not self.nvm_shutdown:
            message = self.recv()
            if message == 'shutdown':
                self.nvm_shutdown = True
    def request(self):
        self.nvm_pipe.send('request')
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
            # break on shutdown signal
            if message == 'shutdown':
                self.nvm_shutdown = True
                self.tk_root.destroy()
                return
            # update rasters
            for index in range(len(message)):
                name, value, pattern = message[index]
                width, height = self.history, len(pattern)
                self.rasters[index] = np.roll(self.rasters[index], 1, axis=1)
                self.rasters[index][:,0] = pattern
                self.images[index] = pil.Image.frombytes('L', (width, height), self.rasters[index].tobytes())
                self.photos[index] = itk.PhotoImage(image=self.images[index])
                self.canvases[index].create_image(1,1,image=self.photos[index], anchor=tk.NW)
                self.labels[index].config(text='%s\n%s'%(name, value))
            # update and request more data
            self.tk_root.update()
            self.request()
        # Queue next update
        self.tk_root.after(5, self.update)
