import numpy as np
import pickle as pk
import matplotlib.pyplot as pt
import os, sys

LABEL_FNAME = "train_dump/labels.pkl"
training_layers = [
    "central_retina_on",
    "central_retina_off",
]
global lr, labels
lr, labels = 0, []

PATCH_FBASE = "train_dump/patches"

def count_examples():
    num_examples =  len([
        name for name in os.listdir('train_dump')
        if os.path.isfile(os.path.join('train_dump', name))])
    num_examples /= 2 # pair of on/off per example
    return num_examples

def label_training_data():
    global labels, lr
    labels = []
    
    num_examples = count_examples()
    print(num_examples)
    
    for ne in range(num_examples):
        print("Example %d"%ne)
        print("Click cross, or left eye followed by right eye.")
        print("Use middle button for cross, left/right buttons for gaze direction.")

        layer_name = training_layers[0]
        activity = np.load("train_dump/%s_%03d.npy"%(layer_name, ne))

        def on_press(event):
            global labels, lr
            done = False
            if event.button == 2:
                feature = "cross"
                done = True
            else:
                feature = ["left eye","right eye"][lr]
                if lr == 1: done = True
                lr += 1
            labels.append({
                "example": ne,
                "feature": feature,
                "position": (int(event.xdata), int(event.ydata)),
                "token": ["left","center","right"][event.button-1]
            })
            with open(LABEL_FNAME,"w") as label_file: pk.dump(labels, label_file)
            print("!!! %s click on %s at %s"%(
                labels[-1]["feature"], labels[-1]["token"], labels[-1]["position"],
            ))
            if done: pt.close() 

        lr = 0
        pt.gcf().canvas.mpl_connect('button_press_event', on_press)
        pt.imshow(activity)
        pt.show()

def show_training_data():
    global labels
    with open(LABEL_FNAME,"r") as label_file: labels = pk.load(label_file)

    for label in labels:

        def on_press(event):
            pt.close()    
        pt.gcf().canvas.mpl_connect('button_press_event', on_press)

        ex = label["example"]
        x, y = label["position"]
        for tl,layer_name in enumerate(training_layers):
            activity = np.load("train_dump/%s_%03d.npy"%(layer_name, ex))
            pt.subplot(1,2,tl+1)
            pt.cla()
            pt.imshow(activity, cmap='gray')
            pt.plot(x,y,'go')
            pt.title(label["feature"] + " : " + label["token"])
        pt.show()

def get_training_patches(rx, ry):
    with open(LABEL_FNAME,"r") as label_file: labels = pk.load(label_file)

    patches = {
        layer_name: {"left": [], "right": [], "center": []}
        for layer_name in training_layers}

    for label in labels:
        ex = label["example"]
        x, y = label["position"]
        for tl,layer_name in enumerate(training_layers):
            activity = np.load("train_dump/%s_%03d.npy"%(layer_name, ex))
            patch = activity[y-ry:y+ry+1,x-rx:x+rx+1]
            if patch.shape != (2*ry+1,2*rx+1): continue
            patches[layer_name][label["token"]].append(patch)

    for layer_name in training_layers:
        for token in ["left","right","center"]:
            patches[layer_name][token] = np.array(
                patches[layer_name][token])
        np.savez(PATCH_FBASE+"_"+layer_name+".npz", **patches[layer_name])
        
def show_training_patches():
    for layer_name in training_layers:
        patch_data = dict(**np.load(PATCH_FBASE+"_"+layer_name+".npz"))
        for token in patch_data:
            print(layer_name,token)
            print(patch_data[token].shape)

if __name__ == "__main__":

    # label_training_data()
    # show_training_data()
    get_training_patches(20,20)
    show_training_patches()
