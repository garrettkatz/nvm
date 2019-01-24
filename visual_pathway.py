from syngen import Network, Environment
from syngen import get_gpus, get_cpu
from syngen import set_suppress_output, set_warnings, set_debug
from syngen import FloatArray
from syngen import create_io_callback

from math import ceil
from os import path
import sys
import argparse
from math import ceil

import numpy as np
from saccade_programs import make_saccade_nvm
from nvm_to_syngen import make_syngen_network, make_syngen_environment, init_syngen_nvm
from syngen import ConnectionFactory, FloatArray
from syngen import get_dsst_params
from oculomotor_nvm import build_network as build_om_network
from oculomotor_nvm import build_environment as build_om_environment
from oculomotor_nvm import build_bridge_connections as build_om_bridge_connections

import matplotlib.pyplot as plt

def build_vp_network(rows, cols):

    center_rows = rows/3
    center_cols = cols/3
    inter_reduction = 11
    inter_rows = int(ceil(float(center_rows)/inter_reduction))
    inter_cols = int(ceil(float(center_cols)/inter_reduction))

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
            "rows" : center_rows,
            "columns" : center_cols,
            "init config": {
                "type": "flat",
                "value": vp_params["cb"].astype(np.float64)[feature]
            }})

        # Add intermediate max layer
        layers.append({
            "name" : "inter_max-%d"%feature,
            "neural model" : "vp_max",
            "rows" : inter_rows,
            "columns" : inter_cols, })

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
    def build_inter_max_weights(from_layer, to_layer, props):
        props["name"] = "%s <- %s"%(to_layer, from_layer)
        props["from layer"] = from_layer
        props["to layer"] = to_layer
        props["type"] = "convergent"
        props["convolutional"] = True
        props["opcode"] = "pool"
        max_index = int(from_layer[-1])
        props["arborized config"] = {
            "field size" : inter_reduction,
            "offset" : 0,
            "stride" : inter_reduction,
            "wrap" : False }
    def build_max_weights(from_layer, to_layer, props):
        props["name"] = "%s <- %s"%(to_layer, from_layer)
        props["from layer"] = from_layer
        props["to layer"] = to_layer
        props["type"] = "subset"
        props["opcode"] = "pool"
        max_index = int(from_layer[-1])
        props["subset config"] = {
            "from row start" : 0,
            "from row end" : inter_rows,
            "from column start" : 0,
            "from column end" : inter_cols,
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
    inter_max_weights_factory = ConnectionFactory(defaults, build_inter_max_weights)
    class_weights_factory = ConnectionFactory(defaults, build_class_weights)
    class_biases_factory = ConnectionFactory(defaults, build_class_biases)

    # Build connections
    connections = []
    for feature in range(5):
        connections.append(inter_max_weights_factory.build(
            "conv-%d"%(feature),
            "inter_max-%d"%feature))
        connections.append(max_weights_factory.build(
            "inter_max-%d"%(feature),
            "max"))
    connections.append(class_weights_factory.build(
        "max","class"))
    connections.append(class_biases_factory.build(
        "class-bias","class"))

    # Build main structure
    structure = {
        "name" : "visual pathway",
        "type" : "parallel",
        "layers" : layers}

    return structure, connections

def build_emot_network(noise=True):
    layers = []
    connections = []

    # Add amygdala layer
    layers.append({
        "name" : "amygdala",
        "neural model" : "oscillator",
        "tau" : 0.04,
        "decay" : 0.04,
        "rows" : 1,
        "columns" : 1
    })

    if noise:
        layers[-1]["init config"] = {
            "type": "normal",
            "mean": 0.0,
            "std dev": 0.025
        }

    # Add vmpfc layer
    layers.append({
        "name" : "vmpfc",
        "neural model" : "oscillator",
        "tau" : 0.01,
        "decay" : 0.01,
        "rows" : 1,
        "columns" : 1,
        "dendrites" : [
            {
                "name" : "gated",
            }
        ]
    })

    if noise:
        layers[-1]["init config"] = {
            "type": "normal",
            "mean": 0.0,
            "std dev": 0.025
        }

    # Gating node for LPFC -> vmPFC
    layers.append({
        "name" : "emot-gate",
        "neural model" : "oscillator",
        "tau" : 0.01,
        "decay" : 0.01,
        "tonic" : 0.0,
        "rows" : 1,
        "columns" : 1,
    })

    # Build main structure
    structure = {
        "name" : "emotional",
        "type" : "parallel",
        "layers" : layers}

    # amygdala -> vmpfc
    # Implements amygdala -> vmpfc mapping for cognitive suppression
    connections.append({
        "name" : "vmpfc <- amygdala",
        "from layer" : "amygdala",
        "to layer" : "vmpfc",
        "type" : "fully connected",
        "opcode" : "sub",
        "plastic" : False,
        "weight config" : {
            "type" : "flat",
            "weight" : 1,
        }})

    # vmpfc -> amygdala
    # Implements vmpfc -> amygdala mapping for interrupt suppression
    connections.append({
        "name" : "amygdala <- vmpfc",
        "from layer" : "vmpfc",
        "to layer" : "amygdala",
        "type" : "fully connected",
        "opcode" : "sub",
        "plastic" : False,
        "weight config" : {
            "type" : "flat",
            "weight" : 1,
        }})

    return structure, connections

def build_vp_bridge_connections(gi_gate):

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

    # bias -> gi
    # Keeps gi in the quiet state
    connections.append({
        "name" : "gi <- bias",
        "from structure" : "nvm",
        "from layer" : "bias",
        "to structure" : "nvm",
        "to layer" : "gi",
        "type" : "fully connected",
        "opcode" : "add",
        "plastic" : False,
        "weight config" : {
            "type" : "flat",
            "weight" : 0,
        }})

    # tc -> amygdala
    # Implements tc -> amygdala mapping for emotional activation
    connections.append({
        "name" : "amygdala <- tc",
        "from structure" : "nvm",
        "from layer" : "tc",
        "to structure" : "emotional",
        "to layer" : "amygdala",
        "type" : "fully connected",
        "opcode" : "add",
        "plastic" : False,
        "weight config" : {
            "type" : "flat",
            "weight" : 0,
        }})

    # tc -> vmPFC
    # Implements tc -> vmPFC mapping for emotional regulation activation
    connections.append({
        "name" : "vmpfc <- tc",
        "from structure" : "nvm",
        "from layer" : "tc",
        "to structure" : "emotional",
        "to layer" : "vmpfc",
        #"dendrite" : "gated",
        "type" : "fully connected",
        "opcode" : "add",
        "plastic" : False,
        "weight config" : {
            "type" : "flat",
            "weight" : 0,
        }})

    # ip -> vmPFC
    # Implements lPFC -> vmPFC mapping for emotional regulation activation
    connections.append({
        "name" : "vmpfc <- ip",
        "from structure" : "nvm",
        "from layer" : "ip",
        "to structure" : "emotional",
        "to layer" : "vmpfc",
        "dendrite" : "gated",
        "type" : "fully connected",
        "opcode" : "add",
        "plastic" : False,
        "weight config" : {
            "type" : "flat",
            "weight" : 0,
        }})

    # vmPFC input gating
    connections.append({
        "name" : "vmpfc <- emot-gate",
        "from layer" : "emot-gate",
        "to layer" : "vmpfc",
        "dendrite" : "gated",
        "type" : "one to one",
        "opcode" : "mult",
        "plastic" : False,
        "weight config" : {
            "type" : "flat",
            "weight" : 1,
        }})

    # go -> vmPFC
    # Implements lpfc -> vmpfc gating to shut down amygdala interrupt
    connections.append({
        "name" : "emot-gate <- go",
        "from structure" : "nvm",
        "from layer" : "go",
        "to structure" : "emotional",
        "to layer" : "emot-gate",
        "type" : "subset",
        "opcode" : "add",
        "plastic" : False,
        "subset config" : {
            "from row start" : 0,
            "from row end" : 1,
            "from column start" : gi_gate,
            "from column end" : gi_gate+1,
            "to row start" : 0,
            "to row end" : 1,
            "to column start" : 0,
            "to column end" : 1
        },
        "weight config" : {
            "type" : "flat",
            "weight" : 1
        }})

    # amygdala -> gi
    # Implements amygdala -> gi mapping for nvm interrupt
    connections.append({
        "name" : "gi <- amygdala",
        "from structure" : "emotional",
        "from layer" : "amygdala",
        "to structure" : "nvm",
        "to layer" : "gi",
        "type" : "fully connected",
        "opcode" : "add",
        "plastic" : False,
        "weight config" : {
            "type" : "flat",
            "weight" : 0,
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

emot_activ = []
emot_bold = []

def build_emot_environment(visualizer=False):
    global emot_bold, emot_activ

    modules = []

    track = ("amygdala", "vmpfc", "emot-gate")
    emot_activ = [[] for _ in xrange(len(track))]

    def track_activ(ID, size, ptr):
        global emot_activ
        emot_activ[ID].append(sum(FloatArray(size, ptr).to_list()))

    create_io_callback("track_activ", track_activ)

    modules.append({
        "type" : "callback",
        "layers" : [
            {
                "layer" : layer_name,
                "output" : True,
                "function" : "track_activ",
                "id" : i
            } for i,layer_name in enumerate(track)
        ]
    })

    if visualizer:
        modules.append({
            "type" : "visualizer",
            "layers" : [
                {"layer" : "vmpfc" },
                {"layer" : "amygdala" },
                {"layer" : "class" }
            ]
        })

    emot_bold = [[] for _ in xrange(len(track))]

    def track_bold(ID, size, ptr):
        global emot_bold
        emot_bold[ID].append(sum(FloatArray(size, ptr).to_list()))

    create_io_callback("track_bold", track_bold)

    modules.append({
        "type" : "callback",
        "layers" : [
            {
                "layer" : layer_name,
                "output" : True,
                "function" : "track_bold",
                "id" : i,
                "output keys" : ["bold"]
            } for i,layer_name in enumerate(track)
        ]
    })

    return modules


def init_vpnet(net, nvmnet, amygdala_sensitivity=0.1, vmpfc_sensitivity=0.1,
        amy_vmpfc_strength=1.0, vmpfc_amy_strength=1.0, lpfc_weight=2.0):
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

    # bias to gi
    # This causes default pattern of gi to be quiet
    mat_name = "gi <- bias"
    mat = net.get_weight_matrix(mat_name)
    gi = nvmnet.layers["gi"]

    quiet = gi.activator.g(gi.coder.encode("quiet"))
    for m in range(mat.size):
        mat.data[m] = quiet.flat[m]

    # tc to amygdala
    # This allows tc to activate amygdala when a face is detected
    mat_name = "amygdala <- tc"
    mat = net.get_weight_matrix(mat_name)

    w = (2.0 * amygdala_sensitivity / 1024) * \
        (tc.coder.encode("left") + tc.coder.encode("right") - tc.coder.encode("cross"))
    for m in range(mat.size):
        mat.data[m] = w[m]

    # tc to vmpfc
    # This allows tc to activate vmpfc when a face is detected
    mat_name = "vmpfc <- tc"
    mat = net.get_weight_matrix(mat_name)

    w = (2.0 * vmpfc_sensitivity / 1024) * \
        (tc.coder.encode("left") + tc.coder.encode("right") - tc.coder.encode("cross"))
    for m in range(mat.size):
        mat.data[m] = w[m]

    # ip to vmpfc
    # This allows tc to activate vmpfc when a face is detected
    mat_name = "vmpfc <- ip"
    mat = net.get_weight_matrix(mat_name)

    # Since ip patterns are 50% on, multiply by 2, divide by layer size
    w = 2.0 * lpfc_weight / 1024
    for m in range(mat.size):
        mat.data[m] = w

    # amygdala to vmPFC
    # This allows the amgydala to activate the vmPFC for reciprocal inhibition
    mat_name = "vmpfc <- amygdala"
    mat = net.get_weight_matrix(mat_name)

    for m in range(mat.size):
        mat.data[m] = amy_vmpfc_strength

    # vmPFC to amygdala
    # This allows the vmPFC to shut down the amygdala
    mat_name = "amygdala <- vmpfc"
    mat = net.get_weight_matrix(mat_name)

    for m in range(mat.size):
        mat.data[m] = vmpfc_amy_strength

    # amygdala to gi
    # This allows amygdala to interrupt nvm
    mat_name = "gi <- amygdala"
    mat = net.get_weight_matrix(mat_name)
    gi = nvmnet.layers["gi"]

    w = np.zeros((gi.size, 1))
    w = gi.coder.encode("pause").reshape((gi.size, 1))
    w = gi.activator.g(w) * 10
    for m in range(mat.size):
        mat.data[m] = w.flat[m]

def main(read=True, visualizer=False, device=None, rate=0, iterations=1000000,
         amygdala_sensitivity=0.0, vmpfc_sensitivity=0.0, amy_vmpfc_strength=1.0,
         vmpfc_amy_strength=1.0, lpfc_weight=2.0, filename="", num_faces=34, noise=True):
    np.set_printoptions(linewidth=200, formatter = {'float': lambda x: '% .2f'%x})
    task = "saccade"
    rows = 340
    cols = 480
    scale = 20

    ''' Build oculomotor '''
    om_network = build_om_network(rows, cols, scale)
    om_modules = build_om_environment(rows, cols, scale, visualizer, task=task,
        train_dump=False, num_faces=num_faces)

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

    ''' Emotional network '''
    emot_structure, emot_connections = build_emot_network(noise)
    emot_modules = build_emot_environment(visualizer)

    ''' entire network '''
    connections = nvm_connections + om_network["connections"] + vp_connections + emot_connections
    motor_gate = nvmnet.gate_map.get_gate_index(('sc', 'op2', 'u'))
    connections += build_om_bridge_connections(motor_gate, rows, cols, scale)

    gi_gate = nvmnet.gate_map.get_gate_index(('di', 'di', 'u'))
    connections += build_vp_bridge_connections(gi_gate)

    net = Network({
        "structures" : [nvm_structure, vp_structure, emot_structure] + om_network["structures"],
        "connections" : connections})

    ''' Filter empty modules '''
    modules = [m
        for m in nvm_modules + om_modules + vp_modules + emot_modules
            if len(m["layers"]) > 0]
    env = Environment({"modules" : modules})

    init_syngen_nvm(nvmnet, net)

    # Draw amygdala sensitivity from distribution centered on parameter
    print("Using amygdala sensitivity: %f" % amygdala_sensitivity)
    print("Using vmPFC sensitivity: %f" % vmpfc_sensitivity)
    print("Using amygdala->vmPFC: %f" % amy_vmpfc_strength)
    print("Using vmPFC->amygdala: %f" % vmpfc_amy_strength)
    print("Using lPFC->vmPFC: %f" % lpfc_weight)
    init_vpnet(net, nvmnet,
        amygdala_sensitivity, vmpfc_sensitivity,
        amy_vmpfc_strength, vmpfc_amy_strength, lpfc_weight)

    ''' Run Simulation '''
    if device is None:
        gpus = get_gpus()
        device = gpus[len(gpus)-1] if len(gpus) > 0 else get_cpu()

    worker_threads = 8 if device == get_cpu() else 0

    report = net.run(env, {"multithreaded" : False,
                               "worker threads" : worker_threads,
                               "devices" : device,
                               "iterations" : iterations,
                               "refresh rate" : rate,
                               "verbose" : True,
                               "learning flag" : True})
    if report is None:
        print("Engine failure.  Exiting...")
        return
    print(report)

    report = report.to_dict()
    print("Response latency mean: %s" % report["layer reports"][0]["Average time"])
    print("Response latency std dev: %s" % report["layer reports"][0]["Standard deviation time"])

    print("")
    print("BOLD Amygdala: %f" % sum(emot_bold[0]))
    print("BOLD vmPFC:    %f" % sum(emot_bold[1]))
    print("BOLD emot-gate:%f" % sum(emot_bold[2]))

    print("")
    print("Activation Amygdala: %f" % sum(emot_activ[0]))
    print("Activation vmPFC:    %f" % sum(emot_activ[1]))
    print("Activation emot-gate:%f" % sum(emot_activ[2]))
    print("Max Activation Amygdala: %f" % max(emot_activ[0]))
    print("Max Activation vmPFC:    %f" % max(emot_activ[1]))
    print("Max Activation emot-gate:%f" % max(emot_activ[2]))

    plt.subplot(211)
    plt.plot(emot_bold[0])
    plt.plot(emot_bold[1])
    plt.plot(emot_bold[2])
    plt.title("BOLD")
    plt.legend(['amy', 'vmPFC', 'emot-gate'], loc='upper left')
    plt.subplot(212)
    plt.plot(emot_activ[0])
    plt.plot(emot_activ[1])
    plt.plot(emot_activ[2])
    plt.title("Activation")
    plt.legend(['amy', 'vmPFC', 'emot-gate'], loc='upper left')

    if filename != "":
        plt.tight_layout()
        plt.savefig(filename)
    else:
        plt.show()

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
    parser.add_argument('-amy_s', type=float, default=0.0,
                        help='amygdala sensitivity')
    parser.add_argument('-vmpfc_s', type=float, default=0.0,
                        help='vmpfc sensitivity')
    parser.add_argument('-av', type=float, default=0.0,
                        help='amygdala -> vmPFC')
    parser.add_argument('-va', type=float, default=0.0,
                        help='vmPFC -> amygdala')
    parser.add_argument('-l', type=float, default=0.0,
                        help='lPFC -> vmPFC')
    parser.add_argument('-f', type=str, default="",
                        help='BOLD/activ filename')
    parser.add_argument('-n', type=int, default=34,
                        help='Number of faces to present')
    parser.add_argument('-noise', action='store_true', default=False,
                        help='amygdala/vmPFC noise')
    args = parser.parse_args()

    if args.host or len(get_gpus()) == 0:
        device = get_cpu()
    else:
        device = get_gpus()[args.d]

    set_suppress_output(False)
    set_warnings(False)
    set_debug(False)

    read = False

    main(read, args.visualizer, device, args.r, 1000000,
        args.amy_s, args.vmpfc_s, args.av, args.va, args.l, args.f, args.n, args.noise)
