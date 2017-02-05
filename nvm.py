import multiprocessing as mp
import matplotlib.pyplot as plt
import Tkinter as tk
import PIL as pil
import PIL.ImageTk as itk # on fedora 24 needed $ dnf install python-pillow-tk
import numpy as np
import time

class nvm:
    def __init__(self):
        self.pipe = mp.Pipe()
        self.process = mp.Process(target=nop, args=(self.pipe[1],))
        self.process.start()
    def f(self, message):
        self.pipe[0].send(message)

def nop(p):
    print('waiting...')
    while True:
        message = p.recv()
        print(message)
        if message == 'q':
            break
    p.close()

class nvm_gui:
    def __init__(self, pipe):
        self.pipe = pipe
        self.block = 255
        self.width, self.height = 512, 256
        self.root = tk.Tk()
        self.frame = tk.Frame(self.root, width=self.width, height=self.height)
        self.frame.pack()
        self.canvas = tk.Canvas(self.frame, width=self.width,height=self.height)
        self.canvas.create_rectangle(0,0,self.width,self.height,fill="red")
        self.canvas.place(x=0,y=0)
        self.root.after(0,self.tick)
        self.root.mainloop()
    def tick(self):
        if self.pipe.poll():
            message = self.pipe.recv()
            # self.block = 255 - self.block
            self.block = message
        # arr = 255*np.ones((4,self.height,self.width),dtype=np.uint8)
        arr = np.random.randint(0,256,(4,self.height,self.width),dtype=np.uint8)
        arr[:, self.height/4:self.height/2, self.width/8:self.width/4] = self.block
        # self.img = pil.Image.frombytes('RGBX', (self.width, self.height), arr.tobytes())
        self.img = pil.Image.frombytes('L', (self.width, self.height), arr[0,:,:].tobytes())
        self.photo = itk.PhotoImage(image=self.img)
        self.canvas.create_image(0,0,image=self.photo, anchor=tk.NW)
        self.root.update()
        self.root.after(10,self.tick)

def f(pipe):
    print("hey")
    ng = nvm_gui(pipe)
    print("heyo")

if __name__ == '__main__':
    main_pipe, ng_pipe = mp.Pipe()
    p = mp.Process(target=f, args=(ng_pipe,))
    print("hi")
    p.start()
    while True:
        message = raw_input('>')
        if message == 'q': break
        main_pipe.send(int(message))
    p.join()
    print("hiya")

    # m = nvm()
    # # time.sleep(1)
    # # m.f('test')
    # while True:
    #     message = raw_input('>')
    #     m.f(message)
    #     if message == 'q':
    #         break
    # m.process.join()

    # height, width = 256, 512
    # # plt.ion()
    # plt.figure()
    # plt.show()
    # for i in range(1000):
    #     arr = np.random.randint(0,256,(height,width,3),np.uint8)
    #     plt.imshow(arr)
    #     plt.draw()
    #     raw_input('')
    # #     # break
    # #     #time.sleep(1)
