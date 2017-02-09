import multiprocessing as mp
import Tkinter as tk
import PIL as pil
import PIL.ImageTk as itk # on fedora 24 needed $ dnf install python-pillow-tk
import numpy as np
import time

class NVMViz:
    def __init__(self, pipe_to_nvm, history=512, padding=16):
        self.pipe_to_nvm = pipe_to_nvm
        self.pipe_to_nvm.send('get state')
        self.history = history
        self.padding = padding
        self.layer_labels = []
        self.canvases = []
        self.rasters = []
        self.imgs = []
        self.photos = []
        layer_size_sum = 0
        nvm_state = self.pipe_to_nvm.recv()
        for layer_index in range(len(nvm_state)):
            layer = nvm_state[layer_index][2]
            layer_size = len(layer)
            layer_size_sum += layer_size
        self.root = tk.Tk()
        self.label_padding = 40
        frame_width = history+2*padding+self.label_padding
        frame_height = layer_size_sum+(len(nvm_state)+1)*padding
        self.frame = tk.Frame(self.root, width = frame_width, height = frame_height)
        layer_size_sum = 0
        for layer_index in range(len(nvm_state)):
            activations = nvm_state[layer_index][2]
            layer_size = len(activations)
            label = tk.Label(self.frame, text=nvm_state[layer_index][0], justify=tk.LEFT)
            label.place(x=0,y=((layer_index+1)*self.padding + layer_size_sum))
            self.layer_labels.append(label)
            width, height = self.history, layer_size
            canvas = tk.Canvas(self.frame, width=width, height=height)
            canvas.place(x=self.padding+self.label_padding,y=((layer_index+1)*self.padding + layer_size_sum))
            self.canvases.append(canvas)
            raster = np.zeros((height, width), dtype=np.uint8)
            self.rasters.append(raster)
            self.imgs.append(None)
            self.photos.append(None)
            layer_size_sum += layer_size
        self.frame.pack()
        self.pipe_to_nvm.send('ready')
        self.root.after(0,self.tick)
        self.root.mainloop()
    def tick(self):
        if self.pipe_to_nvm.poll():
            message = self.pipe_to_nvm.recv()
            if message == 'q':
                self.root.destroy()
                return
            nvm_state = message
            layer_size_sum = 0
            for layer_index in range(len(nvm_state)):
                activations = nvm_state[layer_index][2]
                layer_size = len(activations)
                width, height = self.history, layer_size
                self.rasters[layer_index] = np.roll(self.rasters[layer_index], 1, axis=1)
                activations = np.array(activations, dtype=np.uint8)
                self.rasters[layer_index][:,0] = activations
                self.imgs[layer_index] = pil.Image.frombytes('L', (width, height), self.rasters[layer_index].tobytes())
                self.photos[layer_index] = itk.PhotoImage(image=self.imgs[layer_index])
                self.canvases[layer_index].create_image(1,1,image=self.photos[layer_index], anchor=tk.NW)
                self.layer_labels[layer_index].config(text='%s\n%s'%(nvm_state[layer_index][0], nvm_state[layer_index][1]))
                layer_size_sum += layer_size
            self.pipe_to_nvm.send('ready for up-to-date state')
            self.root.update()
        # print('gui clock: ',time.time())
        self.root.after(5,self.tick)

# def _run_gui(pipe_to_nvm):
#     vmg = VMGui(pipe_to_nvm, history = 512)

# def _run_vm(vm_pipe_to_main):
#     vm_pipe_to_gui, pipe_to_nvm = mp.Pipe()
#     gui_process = mp.Process(target=_run_gui, args=(pipe_to_nvm,))
#     gui_process.start()
#     layer_size = 16
#     while True:
#         if vm_pipe_to_gui.poll():
#             vm_pipe_to_gui.recv()
#             # print(message)
#             state = (
#                 ('{0}', tuple(np.random.randint(0,256,(layer_size,),dtype=np.uint8))),
#                 ('{1}', tuple(np.random.randint(0,256,(layer_size,),dtype=np.uint8))),
#                 ('{ip}', tuple(np.random.randint(0,256,(layer_size,),dtype=np.uint8))),
#             )
#             vm_pipe_to_gui.send(state)
#         if vm_pipe_to_main.poll():
#             message = vm_pipe_to_main.recv()
#             vm_pipe_to_gui.send('q')
#             # print(message)
#             break
#     gui_process.join()

# def q():
#     main_pipe_to_vm.send('q')
#     vm_process.join()
#     quit()

# if __name__ == '__main__':
#     vm_pipe_to_main, main_pipe_to_vm = mp.Pipe()
#     vm_process = mp.Process(target=_run_vm, args=(vm_pipe_to_main,))
#     vm_process.start()
