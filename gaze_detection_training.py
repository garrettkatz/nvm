import numpy as np
import pickle as pk
import matplotlib.pyplot as pt
import os, sys

TRAIN_DIR = "resources/train_dump"
LABEL_FNAME = TRAIN_DIR+"/labels.pkl"
PATCH_FBASE = TRAIN_DIR+"/patches"
TOKENS = ["left", "center", "right"]

TRAINING_LAYERS = [
    "central_retina_on",
    "central_retina_off",
]
global lr, labels
lr, labels = 0, []


def count_examples():
    num_examples =  len([
        name for name in os.listdir(TRAIN_DIR)
        if os.path.isfile(os.path.join(TRAIN_DIR, name))])
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

        layer_name = TRAINING_LAYERS[0]
        activity = np.load(TRAIN_DIR+"/%s_%03d.npy"%(layer_name, ne))

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
                "token": TOKENS[event.button-1]
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
        for tl,layer_name in enumerate(TRAINING_LAYERS):
            activity = np.load(TRAIN_DIR+"/%s_%03d.npy"%(layer_name, ex))
            pt.subplot(1,2,tl+1)
            pt.cla()
            pt.imshow(activity, cmap='gray')
            pt.plot(x,y,'go')
            pt.title(label["feature"] + " : " + label["token"])
        pt.show()

def get_training_patches(rx, ry):
    with open(LABEL_FNAME,"r") as label_file: labels = pk.load(label_file)

    patches = {
        layer_name: {"left": [], "center": [], "right": []}
        for layer_name in TRAINING_LAYERS}

    for label in labels:
        ex = label["example"]
        x, y = label["position"]
        for tl,layer_name in enumerate(TRAINING_LAYERS):
            activity = np.load(TRAIN_DIR+"/%s_%03d.npy"%(layer_name, ex))
            patch = activity[y-ry:y+ry+1,x-rx:x+rx+1]
            if patch.shape != (2*ry+1,2*rx+1): continue
            patches[layer_name][label["token"]].append(patch)

    for layer_name in TRAINING_LAYERS:
        for token in TOKENS:
            patches[layer_name][token] = np.array(
                patches[layer_name][token])
        np.savez(PATCH_FBASE+"_"+layer_name+".npz", **patches[layer_name])

def load_patches():
    return {
        layer_name: dict(**np.load(PATCH_FBASE+"_"+layer_name+".npz"))
        for layer_name in TRAINING_LAYERS}

def show_training_patches():
    patches = load_patches()
    col = 0
    num_rows = 8
    for token in TOKENS:
        for layer_name in TRAINING_LAYERS:
            for row in range(num_rows):
                pt.subplot(num_rows, 6, row*6 + col + 1)
                pt.imshow(patches[layer_name][token][row,:,:])
                if row == 0: pt.title(token)
            col += 1
    pt.show()

def do_patch_training():
    patches = load_patches()
    from sklearn.svm import LinearSVC
    from sklearn.datasets import make_classification

    # format training data
    X, y = [], []
    for t,token in enumerate(TOKENS):
        X_t = np.concatenate(
            [patches[layer_name][token] for layer_name in TRAINING_LAYERS],
            axis = 2
        )
        X_t = X_t.reshape((X_t.shape[0], X_t.shape[1]*X_t.shape[2]))
        y_t = t * np.ones(X_t.shape[0])
        X.append(X_t)
        y.append(y_t)
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    # train one vs rest classifier
    clf = LinearSVC(random_state=0)
    clf.fit(X, y)
    print(clf.coef_.shape)
    print(clf.intercept_.shape)

    # measure training error
    # y_p = clf.predict(X)
    y_p = (X.dot(clf.coef_.T) + clf.intercept_[np.newaxis,:]).argmax(axis=1)
    print("Training error: %d of %d"%((y_p != y).sum(), len(y)))
    
    # visualize kernels
    kernel_shape = patches[TRAINING_LAYERS[0]][TOKENS[0]][0].shape
    kernel_shape = (kernel_shape[0], kernel_shape[1]*2)
    for t, token in enumerate(TOKENS):
        kernel = clf.coef_[t,:].reshape(kernel_shape)
        pt.subplot(len(TOKENS),1,t+1)
        pt.imshow(kernel)
        pt.title(token)
    pt.show()

    # save kernels
    kernels = {}
    for t, token in enumerate(TOKENS):
        k = clf.coef_[t,:].reshape(kernel_shape)
        kernels["%s_%s_w"%(TRAINING_LAYERS[0], token)] = k[:,:kernel_shape[0]]
        kernels["%s_%s_w"%(TRAINING_LAYERS[1], token)] = k[:,kernel_shape[0]:]
    kernels["b"] = clf.intercept_
    np.savez(TRAIN_DIR+"/kernels.npz", **kernels)

def do_central_vision_testing():
    from scipy.signal import correlate2d

    kernels = dict(**np.load(TRAIN_DIR+"/kernels.npz"))
    with open(LABEL_FNAME,"r") as label_file: labels = pk.load(label_file)

    for label in labels:

        def on_press(event):
            pt.close()    
        pt.gcf().canvas.mpl_connect('button_press_event', on_press)

        ex = label["example"]
        x, y = label["position"]
        filtered = {}
        for tl,layer_name in enumerate(TRAINING_LAYERS):
            activity = np.load(TRAIN_DIR+"/%s_%03d.npy"%(layer_name, ex))
            pt.subplot(4,2,tl+1)
            pt.cla()
            pt.imshow(activity, cmap='gray')
            pt.plot(x,y,'go')
            pt.title(label["feature"] + " : " + label["token"])
            
            for t, token in enumerate(TOKENS):
                k = kernels["%s_%s_w"%(layer_name, token)]
                filtered[layer_name,token] = correlate2d(activity, k, mode='same')
                pt.subplot(4, 2, 2*(t+1) + tl + 1)
                pt.cla()
                pt.imshow(filtered[layer_name,token], cmap='gray')
                pt.title(token)

        for t,token in enumerate(TOKENS):
            filt = filtered[TRAINING_LAYERS[0],token]
            filt += filtered[TRAINING_LAYERS[1],token]
            iidx = filt.argmax(axis=0)
            imax = filt.max(axis=0)
            j = imax.argmax()
            i = iidx[j]
            # print(filt.max(), filt[i,j])
            score = filt.max() + kernels["b"][t]
            print("%s score: %f (%f)"%(token, score, kernels["b"][t]))
            pt.subplot(4,2,1)
            pt.plot(j,i,'ro')

        pt.tight_layout()
        pt.show()

def do_cnn_training(show=False):

    import torch
    from torch.autograd import Variable
    import torch.nn as nn
    import torch.nn.functional as F
    
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            in_channels = 2 # 1 each for on/off retinal layers
            out_channels = 3 # 3 tokens
            kernel_size = (41,41) # (height, width) of kernel
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
    
        def forward(self, x):
            x = self.conv(x)
            # x, _ = torch.max(x,3)
            # x, _ = torch.max(x,2)
            return x
    
    net = Net()

    # initialize with kernels instead of random
    kernels = dict(**np.load(TRAIN_DIR+"/kernels.npz"))
    w0 = np.zeros((3,2,41,41), dtype=np.float32)
    for t, token in enumerate(TOKENS):
        for tl, layer in enumerate(TRAINING_LAYERS):
            w0[t,tl,:,:] = kernels["%s_%s_w"%(layer, token)]
    net.conv.weight.data = torch.from_numpy(w0.copy())
    net.conv.bias.data = torch.from_numpy(kernels["b"].astype(np.float32))

    with open(LABEL_FNAME,"r") as label_file: labels = pk.load(label_file)
    X = []
    Z = []
    Y = []
    rows, columns = 113-40, 160-40
    R, C = np.mgrid[:rows,:columns] # transpose for bitmap
    for label in labels:
        ex = label["example"]
        token = label["token"]
        col,row = label["position"]

        if label["feature"] in ["cross", "left eye"]:
            x = np.array([
                np.load(TRAIN_DIR+"/%s_%03d.npy"%(layer_name, ex))
                for layer_name in TRAINING_LAYERS])
            y = np.array([
                1 if t == token else -1 for t in TOKENS])
            z = np.zeros((len(TOKENS), rows, columns), dtype=np.float32)

        t = TOKENS.index(token)
        z[t,:,:] += np.exp(-((C-(col-20.))**2 + (R-(row-20.))**2)/(21)**2)

        if label["feature"] in ["cross", "right eye"]:
            X.append(x)
            Y.append(y)
            Z.append(z)

    X = np.array(X).astype(np.float32)
    Y = np.array(Y).astype(np.float32)
    Z = np.array(Z).astype(np.float32)

    criterion = nn.MSELoss()
    exs = range(X.shape[0])
    pt.ion()

    net.zero_grad()
    for epoch in range(200):
        net_loss = 0.
        for ex in exs:
            input_pattern = X[ex][np.newaxis,:,:,:]
            input_pattern = Variable(torch.from_numpy(input_pattern))
            z = net.forward(input_pattern)
            # loss = criterion(y, Variable(torch.from_numpy(Y[ex])))
            loss = criterion(z, Variable(torch.from_numpy(Z[ex])))
            loss.backward()
            # print("%d,%d: loss %f"%(
            #     epoch, ex, loss.data.numpy()[0]))
            net_loss += loss.data.numpy()[0]
            # print("%d,%d: loss %f, %s ~ %s"%(
            #     epoch, ex, loss.data.numpy()[0], y.data.numpy(), Y[ex]))
            # print(y.data.numpy())
            # print(Y[ex])

        learning_rate = 0.0001
        for f in net.parameters():
            f.data.sub_(f.grad.data * learning_rate)
        net.zero_grad()

        w = net.conv.weight.data.numpy()
        b = net.conv.bias.data.numpy()
        np.savez("resources/train_dump/cnn_wb.npz", **{"w": w, "b": b})
        if show:
            for c in range(3):
                for t in range(2):
                    pt.subplot(3,4,4*c+t+1)
                    pt.cla()
                    pt.imshow(w[c,t,:,:], cmap='gray')
            for c in range(3):
                for t in range(2):
                    pt.subplot(3,4,4*c+t+3)
                    pt.cla()
                    pt.imshow(w0[c,t,:,:], cmap='gray')
            pt.show()
            pt.gcf().canvas.draw()
            pt.gcf().canvas.flush_events()

        print("%d: %f"%(epoch, net_loss))
            
    ex = 10
    input_pattern = X[ex][np.newaxis,:,:,:]
    input_pattern = Variable(torch.from_numpy(input_pattern.astype(np.float32)))
    # z = net.conv(input_pattern).data.numpy()
    z = net.forward(input_pattern).data.numpy()
    print(X[ex].shape)
    print(z.shape)
    w = net.conv.weight.data.numpy()

    if show:
        pt.ioff()
        for c in range(3):
            pt.subplot(3,4,4*c+1)
            pt.cla()
            pt.imshow(z[0,c,:,:], cmap='gray', vmin=0, vmax=1)
            pt.subplot(3,4,4*c+2)
            pt.cla()
            pt.imshow(Z[ex,c,:,:], cmap='gray', vmin=0, vmax=1)
            pt.subplot(3,4,4*c+3)
            pt.cla()
            pt.imshow(w[c,0,:,:], cmap='gray')
            pt.subplot(3,4,4*c+4)
            pt.cla()
            pt.imshow(w[c,1,:,:], cmap='gray')
        pt.tight_layout()
        pt.show()


if __name__ == "__main__":

    # label_training_data()
    # show_training_data()
    # get_training_patches(20,20)
    # show_training_patches()
    # do_patch_training()
    # do_central_vision_testing()
    do_cnn_training(show=False)
