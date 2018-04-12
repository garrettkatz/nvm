import numpy as np
import matplotlib.pyplot as pt
import os, sys

training_layers = [
    "central_retina_on",
    "central_retina_off",
]
global ne, ln
ne, ln = 0, 0

def update_example(ne, ln):
    print("Click right then left or middle(cross)")
    layer_name = training_layers[ln]
    activity = np.load("train_dump/%s_%03d.npy"%(layer_name, ne))
    pt.imshow(activity)

def show_training_data():
    global ne, ln

    # label_file = open("train_dump/coordinates.txt","w")
    label_file = sys.stdout
    label_file.write("object,x,y")

    num_examples =  len([
        name for name in os.listdir('train_dump')
        if os.path.isfile(os.path.join('train_dump', name))])
    num_examples /= 2 # pair of on/off per example
    print(num_examples)    
    
    examples = []
    def on_press(event):
        global ne, ln
        # print("%s click at %d,%d"%(
        #     ["left","middle","right"][event.button-1],
        #     event.xdata, event.ydata,
        # ))
        label_file.write("%s click at %d,%d"%(
            ["left eye","cross","right eye"][event.button-1],
            event.xdata, event.ydata,
        ))
        if event.button < 3:
            ln = (ln + 1) % len(training_layers)
            if ln == 0: ne += 1
            if ne < num_examples: update_example(ne, ln)
            else: print("DONE.")

    pt.ion()
    pt.gcf().canvas.mpl_connect('button_press_event', on_press)
    update_example(ln, ne)
    pt.show()

    # for ne in range(num_examples):
    #     examples.append({
    #         layer_name: np.load("train_dump/%s_%03d.npy"%(layer_name, ne))
    #         for layer_name in training_layers})
    #     for layer_name in training_layers:
    #         pt.imshow(examples[-1][layer_name])
    #         pt.show()
            

if __name__ == "__main__":
    show_training_data()
