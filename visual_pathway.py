from syngen import Network, Environment
from syngen import get_gpus, get_cpu
from syngen import set_suppress_output, set_warnings, set_debug
from syngen import FloatArray
from syngen import create_io_callback

from os import path
import sys
import argparse

import numpy as np
from saccade_programs import make_saccade_nvm
from nvm_to_syngen import make_syngen_network, make_syngen_environment, init_syngen_nvm
from syngen import ConnectionFactory, FloatArray
from syngen import get_dsst_params
from oculomotor_nvm import build_network as build_om_network
from oculomotor_nvm import build_environment as build_om_environment
from oculomotor_nvm import build_bridge_connections as build_om_bridge_connections

def build_vp_network(rows, cols):

    # load trained parameters
    vp_params = dict(**np.load("cnn2.npz"))

    # Create layers
    layers = []

    # Add convolutional and sum layers
    for feature in range(5):
        layers.append({
            "name" : "conv-%d"%feature,
            "neural model" : "relay",
            "ramp" : False,
            "rows" : rows/3,
            "columns" : cols/3,
            "init config": {
                "type": "flat",
                "value": vp_params["cb"].astype(np.float64)[feature]
            }})

    # Add max layer
    layers.append({
        "name" : "max",
        "neural model" : "vp_max",
        "rows" : 1,
        "columns" : 5, })
    
    # Add one-hot class layer with bias
    layers.append({
        "name" : "class-bias",
        "neural model" : "relay",
        "ramp" : False,
        "rows" : 1,
        "columns" : 1,
        "init config": {
            "type": "flat",
            "value": 1
        }})
    layers.append({
        "name" : "class",
        "neural model" : "nvm_logistic",
        "rows" : 1,
        "columns" : 3, })
    
    # create internal connection factories
    def build_max_weights(from_layer, to_layer, props):
        props["name"] = "%s <- %s"%(to_layer, from_layer)
        props["from layer"] = from_layer
        props["to layer"] = to_layer
        props["type"] = "subset"
        props["opcode"] = "pool"
        max_index = int(from_layer[-1])
        props["subset config"] = {
            "from row start" : 0,
            "from row end" : rows/3,
            "from column start" : 0,
            "from column end" : cols/3,
            "to row start" : 0,
            "to row end" : 1,
            "to column start" : max_index,
            "to column end" : max_index + 1 }
    def build_class_weights(from_layer, to_layer, props):
        props["name"] = "%s <- %s"%(to_layer, from_layer)
        props["from layer"] = from_layer
        props["to layer"] = to_layer
        props["type"] = "fully connected"
        props["opcode"] = "add"
    def build_class_biases(from_layer, to_layer, props):
        props["name"] = "%s <- %s"%(to_layer, from_layer)
        props["from layer"] = from_layer
        props["to layer"] = to_layer
        props["type"] = "fully connected"
        props["opcode"] = "add"
    # Parameters shared by all connections
    defaults = {
        "from structure" : "visual pathway", "to structure" : "visual pathway",
        "plastic" : False, "sparse" : False,
        "weight config": {"type" : "flat", "weight" : 1 }}
    max_weights_factory = ConnectionFactory(defaults, build_max_weights)
    class_weights_factory = ConnectionFactory(defaults, build_class_weights)
    class_biases_factory = ConnectionFactory(defaults, build_class_biases)

    # Build connections
    connections = []
    for feature in range(5):
        connections.append(max_weights_factory.build(
            "conv-%d"%(feature),
            "max"))
    connections.append(class_weights_factory.build(
        "max","class"))
    connections.append(class_biases_factory.build(
        "class-bias","class"))

    # Build main structure (feedforward engine)
    structure = {
        "name" : "visual pathway",
        "type" : "feedforward",
        "layers" : layers}

    return structure, connections

def build_vp_bridge_connections():

    # Retina -> convolutional
    connections = []
    for feature in range(5):
        for channel in ["on","off"]:
            connections.append({
                "name" : "conv-%s-%d-kernel"%(channel, feature),
                "from structure" : "retina",
                "from layer" : "central_retina_%s"%channel,
                "to structure" : "visual pathway",
                "to layer" : "conv-%d"%feature,
                "type" : "convergent",
                "convolutional" : True,
                "opcode" : "add",
                "plastic" : False,
                "weight config" : {
                    "type" : "flat",
                    "weight" : 1,
                },
                "arborized config" : {
                    "field size" : 41,
                    "stride" : 1,
                    "wrap" : False
                }})

    # class -> temporal cortex
    connections.append({
        "name" : "tc <- class",
        "from structure" : "visual pathway",
        "from layer" : "class",
        "to structure" : "nvm",
        "to layer" : "tc",
        "type" : "fully connected",
        "opcode" : "add",
        "plastic" : False,
        "weight config" : {
            "type" : "flat",
            "weight" : 1,
        }})

    return connections

def build_vp_environment(visualizer=False):

    read_layers = [] #["max","class"]
    modules = []

    modules.append({
        "type" : "callback",
        "layers" : [
            {
                "structure" : "visual pathway",
                "layer" : layer_name,
                "output" : True,
                "function" : "vp_read",
                "id" : i
            } for i,layer_name in enumerate(read_layers)]})

    if visualizer:
        modules.append({
            "type" : "visualizer",
            "layers" : [
                {"layer" : "conv-%d"%(feature) }
                for feature in range(5)
            ] + [{"layer" : l} for l in read_layers]
        })

    def read_callback(ID, size, ptr):

        layer_name = read_layers[ID]
        h = np.array(FloatArray(size,ptr).to_list())
        if len(h) < 10:
            print(layer_name, h)
        else:
            print(layer_name, h.min(), h.max(), h.mean())

    create_io_callback("vp_read", read_callback)
    
    return modules


def init_vpnet(net, nvmnet):
    # load trained parameters
    vp_params = dict(**np.load("cnn2.npz"))

    # convolutional kernels
    for feature in range(5):
        for c,channel in enumerate(["on","off"]):
            kernel_name = "conv-%s-%d-kernel"%(channel, feature)
            kernel = vp_params["cw"][feature, c].astype(np.float64)
            mat = net.get_weight_matrix(kernel_name)
            for m in range(mat.size):
                mat.data[m] = kernel.flat[m]

    # linear class detector
    mat_name = "class <- max"
    mat = net.get_weight_matrix(mat_name)
    for m in range(mat.size):
        mat.data[m] = vp_params["lw"].astype(np.float64).flat[m]
    mat_name = "class <- class-bias"
    mat = net.get_weight_matrix(mat_name)
    for m in range(mat.size):
        mat.data[m] = vp_params["lb"].astype(np.float64).flat[m]

    # class to tc
    mat_name = "tc <- class"
    mat = net.get_weight_matrix(mat_name)
    tc = nvmnet.layers["tc"]
    w = np.concatenate([
        tc.coder.encode(tok) for tok in ["left","cross","right"]
        ], axis=1)
    w = tc.activator.g(w) * 10 # wipe out previous
    for m in range(mat.size):
        mat.data[m] = w.flat[m]

def main(read=True, visualizer=False, device=None, rate=0, iterations=1000000):
    np.set_printoptions(linewidth=200, formatter = {'float': lambda x: '% .2f'%x})
    task = "saccade"
    rows = 340
    cols = 480
    scale = 5

    ''' Build oculomotor '''
    om_network = build_om_network(rows, cols, scale)
    om_modules = build_om_environment(rows, cols, scale, visualizer, task=task, train_dump=False)

    ''' Build NVM '''
    nvmnet = make_saccade_nvm("logistic")
    nvm_structure, nvm_connections = make_syngen_network(nvmnet)
    if visualizer:
        viz_layers = ["sc","fef","tc","ip","opc","op1","op2","gh","go", "ci", "csom", "co"]
    else:
        viz_layers = []
    nvm_modules = make_syngen_environment(nvmnet,
        initial_patterns = dict(nvmnet.activity),
        run_nvm=False,
        viz_layers = viz_layers,
        print_layers = nvmnet.layers,
        stat_layers=[],
        read=read)

    ''' Visual pathway '''
    vp_structure, vp_connections = build_vp_network(rows, cols)
    vp_modules = build_vp_environment(visualizer)

    ''' entire network '''
    connections = nvm_connections + om_network["connections"] + vp_connections
    connections += build_om_bridge_connections()
    connections += build_vp_bridge_connections()

    net = Network({
        "structures" : [nvm_structure, vp_structure] + om_network["structures"],
        "connections" : connections})
    env = Environment({"modules" : nvm_modules + om_modules + vp_modules})

    init_syngen_nvm(nvmnet, net)
    init_vpnet(net, nvmnet)

    ''' Run Simulation '''
    if device is None:
        gpus = get_gpus()
        device = gpus[len(gpus)-1] if len(gpus) > 0 else get_cpu()

    report = net.run(env, {"multithreaded" : True,
                               "worker threads" : "4",
                               "devices" : device,
                               "iterations" : iterations,
                               "refresh rate" : rate,
                               "verbose" : True,
                               "learning flag" : True})
    if report is None:
        print("Engine failure.  Exiting...")
        return
    print(report)

    # Delete the objects
    del net
    del env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', action='store_true', default=False,
                        dest='visualizer',
                        help='run the visualizer')
    parser.add_argument('-host', action='store_true', default=False,
                        help='run on host CPU')
    parser.add_argument('-d', type=int, default=1,
                        help='run on device #')
    parser.add_argument('-r', type=int, default=0,
                        help='refresh rate')
    args = parser.parse_args()

    if args.host or len(get_gpus()) == 0:
        device = get_cpu()
    else:
        device = get_gpus()[args.d]

    set_suppress_output(False)
    set_warnings(False)
    set_debug(False)

    read = True

    main(read, args.visualizer, device, args.r)
