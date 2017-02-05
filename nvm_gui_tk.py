import Tkinter
from PIL import Image, ImageTk
import numpy
import time

class mainWindow():
	times=1
	timestart=time.clock()
	data=numpy.array(numpy.random.random((400,500))*100,dtype=int)

	def __init__(self):
		self.root = Tkinter.Tk()
		self.frame = Tkinter.Frame(self.root, width=500, height=400)
		self.frame.pack()
		self.canvas = Tkinter.Canvas(self.frame, width=500,height=400)
		self.canvas.place(x=-2,y=-2)
		self.root.after(1000,self.start) # INCREASE THE 0 TO SLOW IT DOWN
		self.root.mainloop()

	def start(self):
		global data
		self.im=Image.fromstring('L', (self.data.shape[1],
			self.data.shape[0]), self.data.astype('b').tostring())
		self.photo = ImageTk.PhotoImage(image=self.im)
		self.canvas.create_image(0,0,image=self.photo,anchor=Tkinter.NW)
		self.root.update()
		self.times+=1
		if self.times%33==0:
			print "%.02f FPS"%(self.times/(time.clock()-self.timestart))
		self.root.after(10,self.start)
		self.data=numpy.roll(self.data,-1,1)

# if __name__ == '__main__':
#     x=mainWindow()

import Tkinter as tk
import numpy as np
from PIL import Image, ImageTk

root = tk.Tk()

# label = tk.Label(root, text="Hello!")
# label.pack()

width, height = 512, 256

canvas = tk.Canvas(root, width=width, height=height)
canvas.pack()

# canvas.create_rectangle(0,0,width,height, fill="blue")

arr = 255*np.ones((4,height,width),dtype=np.uint8)
arr[:, height/4:height/2, width/8:width/4] = 0
# arr[:,:,3] = 0
# img = Image.fromarray(arr,mode='RGB')
# pimg = ImageTk.PhotoImage(img)
# pimg = ImageTk.PhotoImage('RGB',(height,width),data=arr.astype('b').tostring())
# img = Image.frombytes('RGB', (height,width), arr.tobytes())
img = Image.frombytes('RGB', (width,height), arr.tobytes())
pimg = ImageTk.PhotoImage(image=img)
# print(img)
# print(pimg)
canvas.create_image(0,0,image=pimg)
root.mainloop()

# from Tkinter import *
# import numpy as np
# import Image
# # import ImageTk
# def callback(event):
#      # do some stuff with a numpy array
#      # ideally, e.g.:
#      x=event.x; y=event.y
#      val=np.asarray(imgTk)[x,y]
#      print val
# arr=np.ones([256,256])
# img=Image.fromarray(arr)
# imgTk=PhotoImage(img)
# t=Tk()
# l=Label(t)
# l.configure(image=imgTk)
# l.bind('<Motion>', callback)
# l.pack()
# t.mainloop()
