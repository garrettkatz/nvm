import multiprocessing as mp
import Tkinter as tk
import PIL as pil
import PIL.ImageTk as itk # on fedora 24 needed $ dnf install python-pillow-tk
import numpy as np
import time

class VMGui:
    def __init__(self, gui_pipe_to_vm, history=512, padding=16):
        self.gui_pipe_to_vm = gui_pipe_to_vm
        self.gui_pipe_to_vm.send('get state')
        self.history = history
        self.padding = padding
        self.layer_labels = []
        self.canvases = []
        self.rasters = []
        self.imgs = []
        self.photos = []
        layer_size_sum = 0
        vm_state = self.gui_pipe_to_vm.recv()
        for ell in range(len(vm_state)):
            layer = vm_state[ell][1]
            layer_size = len(layer)
            layer_size_sum += layer_size
        self.root = tk.Tk()
        self.frame = tk.Frame(self.root, width = history+2*padding+20, height = layer_size_sum+(len(vm_state)+1)*padding)
        layer_size_sum = 0
        for ell in range(len(vm_state)):
            layer = vm_state[ell][1]
            layer_size = len(layer)
            label = tk.Label(self.frame, text=vm_state[ell][0])
            label.place(x=0,y=((ell+1)*self.padding + layer_size_sum))
            width, height = self.history, layer_size
            canvas = tk.Canvas(self.frame, width=width, height=height)
            canvas.place(x=self.padding+20,y=((ell+1)*self.padding + layer_size_sum))
            self.canvases.append(canvas)
            raster = np.zeros((height, width), dtype=np.uint8)
            self.rasters.append(raster)
            self.imgs.append(None)
            self.photos.append(None)
            layer_size_sum += layer_size
        self.frame.pack()
        self.gui_pipe_to_vm.send('ready')
        self.root.after(0,self.tick)
        self.root.mainloop()
    def tick(self):
        if self.gui_pipe_to_vm.poll():
            message = self.gui_pipe_to_vm.recv()
            if message == 'q':
                self.root.destroy()
                return
            vm_state = message
            layer_size_sum = 0
            for ell in range(len(vm_state)):
                layer = vm_state[ell][1]
                layer_size = len(layer)
                width, height = self.history, layer_size
                self.rasters[ell] = np.roll(self.rasters[ell], 1, axis=1)
                tt = np.array(layer)
                self.rasters[ell][:,0] = tt
                self.imgs[ell] = pil.Image.frombytes('L', (width, height), self.rasters[ell].tobytes())
                self.photos[ell] = itk.PhotoImage(image=self.imgs[ell])
                self.canvases[ell].create_image(0,0,image=self.photos[ell], anchor=tk.NW)
                layer_size_sum += layer_size
            self.gui_pipe_to_vm.send('ready for up-to-date state')
            self.root.update()
        # print('gui clock: ',time.time())
        self.root.after(5,self.tick)

def _run_gui(gui_pipe_to_vm):
    vmg = VMGui(gui_pipe_to_vm, history = 512)

def _run_vm(vm_pipe_to_main):
    vm_pipe_to_gui, gui_pipe_to_vm = mp.Pipe()
    gui_process = mp.Process(target=_run_gui, args=(gui_pipe_to_vm,))
    gui_process.start()
    layer_size = 16
    while True:
        if vm_pipe_to_gui.poll():
            vm_pipe_to_gui.recv()
            # print(message)
            state = (
                ('{0}', tuple(np.random.randint(0,256,(layer_size,),dtype=np.uint8))),
                ('{1}', tuple(np.random.randint(0,256,(layer_size,),dtype=np.uint8))),
                ('{ip}', tuple(np.random.randint(0,256,(layer_size,),dtype=np.uint8))),
            )
            vm_pipe_to_gui.send(state)
        if vm_pipe_to_main.poll():
            message = vm_pipe_to_main.recv()
            vm_pipe_to_gui.send('q')
            # print(message)
            break
    gui_process.join()

def q():
    main_pipe_to_vm.send('q')
    vm_process.join()
    quit()

if __name__ == '__main__':
    vm_pipe_to_main, main_pipe_to_vm = mp.Pipe()
    vm_process = mp.Process(target=_run_vm, args=(vm_pipe_to_main,))
    vm_process.start()
