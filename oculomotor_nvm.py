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

dsst_params = get_dsst_params(num_rows=8, cell_res=8)

def build_retina(name, rows, cols, rf_size=5, convergence=1):
    photoreceptor = {
        "name" : name + "_photoreceptor",
        "neural model" : "relay",
        "rows" : rows,
        "columns" : cols,
        "ramp" : True}

    photoreceptor_inverse = {
        "name" : name + "_photoreceptor_inverse",
        "neural model" : "relay",
        "rows" : rows,
        "columns" : cols,
        "ramp" : True,
        "init config" : {
            "type" : "flat",
            "value" : 1.0
        }}

    retina_on = {
        "name" : name + "_retina_on",
        #"neural model" : "rate_encoding",  # use rate_encoding for new visual pathway
        "neural model" : "relay",
        "rows" : rows / convergence,
        "columns" : cols / convergence,
        "ramp" : True}

    retina_off = {
        "name" : name + "_retina_off",
        #"neural model" : "rate_encoding",  # use rate_encoding for new visual pathway
        "neural model" : "relay",
        "rows" : rows / convergence,
        "columns" : cols / convergence,
        "ramp" : True}

    connections = [
        # Photoreceptor -> Photoreceptor Inverse
        {
            "from layer" : name + "_photoreceptor",
            "to layer" : name + "_photoreceptor_inverse",
            "type" : "one to one",
            "opcode" : "sub",
            "plastic" : False,
            "weight config" : {
                "type" : "flat",
                "weight" : 1.0,
            },
        },
        # Photoreceptor -> ON/OFF
        {
            "from layer" : name + "_photoreceptor",
            "to layer" : name + "_retina_on",
            "type" : "convergent",
            "convolutional" : True,
            "opcode" : "add",
            "plastic" : False,
            "weight config" : {
                "type" : "flat",
                "weight" : 1.0,
                "distance callback" : "mexican_hat",
                "to spacing" : convergence,
            },
            "arborized config" : {
                "field size" : rf_size,
                "stride" : convergence,
                "wrap" : True
            }
        },
        {
            "from layer" : name + "_photoreceptor_inverse",
            "to layer" : name + "_retina_off",
            "type" : "convergent",
            "convolutional" : True,
            "opcode" : "add",
            "plastic" : False,
            "weight config" : {
                "type" : "flat",
                "weight" : 1.0,
                "distance callback" : "mexican_hat",
                "to spacing" : convergence,
            },
            "arborized config" : {
                "field size" : rf_size,
                "stride" : convergence,
                "wrap" : True
            }
        }
    ]

    return [photoreceptor, photoreceptor_inverse, retina_on, retina_off], connections


def build_network(rows=200, cols=200, scale=5):
    # Create main structure (parallel engine)
    sc_structure = {"name" : "oculomotor", "type" : "parallel"}
    retina_structure = {"name" : "retina", "type" : "feedforward"}

    # Add retinal layers
    p_retina_layers, p_retina_connections = build_retina(
        "peripheral", rows, cols,
        rf_size = 5,
        convergence = scale)
    c_retina_layers, c_retina_connections = build_retina(
        "central", rows/3, cols/3,
        rf_size = 5,
        convergence = 1)

    # Add superior colliculus layers
    sc_sup = {
        "name" : "sc_sup",
        "neural model" : "rate_encoding",
        "rows" : int(rows / scale),
        "columns" : int(cols / scale)
        }

    sc_deep = {
        "name" : "sc_deep",
        "neural model" : "rate_encoding",
        "rows" : int(rows / scale),
        "columns" : int(cols / scale)}

    gating_layer = {
        "name" : "gating",
        "neural model" : "rate_encoding",
        "rows" : int(rows / scale),
        "columns" : int(cols / scale),
        "dendrites" : [{"name" : "disinhibition"}],
        "init config" : {
            "type" : "flat",
            "value" : 1.0
        }
    }

    # Add layers to structure
    sc_structure["layers"] = [sc_sup, sc_deep, gating_layer]
    retina_structure["layers"] = p_retina_layers + c_retina_layers

    # Create connections
    receptive_field = 5
    sc_input_strength = 1.0
    sc_to_motor_strength = 1.0
    gate_strength = 5.0
    connections = [
        # ON/OFF -> SC
        {
            "from layer" : "peripheral_retina_on",
            "to layer" : "sc_sup",
            "type" : "convergent",
            "convolutional" : True,
            "opcode" : "add",
            "plastic" : False,
            "weight config" : {
                "type" : "flat",
                "weight" : sc_input_strength,
                "distance callback" : "mexican_hat",
            },
            "arborized config" : {
                "field size" : receptive_field,
                "stride" : 1,
                "wrap" : False
            }
        },
        {
            "from layer" : "peripheral_retina_off",
            "to layer" : "sc_sup",
            "type" : "convergent",
            "convolutional" : True,
            "opcode" : "add",
            "plastic" : False,
            "weight config" : {
                "type" : "flat",
                "weight" : sc_input_strength,
                "distance callback" : "mexican_hat",
            },
            "arborized config" : {
                "field size" : receptive_field,
                "stride" : 1,
                "wrap" : False
            }
        },
        # SC -> SC out
        {
            "from layer" : "sc_sup",
            "to layer" : "sc_deep",
            "type" : "convergent",
            "convolutional" : True,
            "opcode" : "add",
            "plastic" : False,
            "weight config" : {
                "type" : "flat",
                "weight" : sc_to_motor_strength,
                "distance callback" : "mexican_hat",
            },
            "arborized config" : {
                "field size" : receptive_field,
                "stride" : 1,
                "wrap" : False
            }
        },
        # Gating -> SC out
        {
            "from layer" : "gating",
            "to layer" : "sc_deep",
            "type" : "one to one",
            "opcode" : "sub",
            "plastic" : False,
            "weight config" : {
                "type" : "flat",
                "weight" : gate_strength
            }
        }
    ]

    # Create network
    return {"structures" : [sc_structure, retina_structure],
         "connections" : connections + p_retina_connections + c_retina_connections}

train_dump_index = 0

def build_environment(rows=200, cols=200, scale=5, visualizer=False,
        task="saccade", train_dump=False, num_faces=34):
    # Create training data callback
    global train_dump_index
    # train_dump_index = 0

    training_layers = [
        "central_retina_on",
        "central_retina_off",
    ]
    def dump_training_data(ID, size, ptr):
        global train_dump_index

        layer_name = training_layers[ID]
        fname = "resources/train_dump/%s_%03d.npy"%(
            layer_name, train_dump_index
        )
        if ID == len(training_layers)-1: train_dump_index += 1

        activity = np.array(FloatArray(size,ptr).to_list())
        activity = activity.reshape((rows/3, cols/3))
        np.save(fname, activity)

    create_io_callback("dump_training_data", dump_training_data)

    # Create environment modules
    if task == "dsst":
        modules = [
            {
                "type" : "dsst",
                "rows" : dsst_params["rows"],
                "columns" : dsst_params["columns"],
                "cell size" : dsst_params["cell columns"],
                "layers" : [
                    {
                        "layer" : "peripheral_photoreceptor",
                        "input" : True,
                    }
                ]
            }
        ]
    elif task == "saccade":
        modules = [
            {
                "type" : "saccade",
                "saccade rate" : 0.75,
                "automatic" : True,
                "shuffle" : True,
                "num faces" : num_faces,
                "cross time" : 1000,
                "face time" : 1000,
                "layers" : [
                    {
                        "layer" : "peripheral_photoreceptor",
                        "input" : True,
                    },
                    {
                        "layer" : "central_photoreceptor",
                        "input" : True,
                        "central" : True,
                    },
                    {
                        "layer" : "sc_deep",
                        "output" : True,
                    }
                ]
            }
        ]
        if train_dump:
            modules.append({
                "type" : "callback",
                "layers" : [
                    {
                        "layer" : layer_name,
                        "output" : True,
                        "function" : "dump_training_data",
                        "id" : i
                    } for i,layer_name in enumerate(training_layers)
                ]
            })
    else:
        modules = [
            {
                "type" : "image_input",
                "filename" : image_filename,
                "layers" : [
                    {
                        "layer" : "peripheral_photoreceptor",
                        "input" : True,
                    }
                ]
            }
        ]

    if visualizer:
        modules.append({
            "type" : "visualizer",
            "layers" : [
                {"layer" : "peripheral_photoreceptor" },
                {"layer" : "peripheral_retina_on" },
                {"layer" : "peripheral_retina_off" },
                {"layer" : "sc_sup" },
                {"layer" : "sc_deep" },
                {"layer" : "gating" },
                {"layer" : "central_photoreceptor" },
                {"layer" : "central_retina_on" },
                {"layer" : "central_retina_off" },
            ]
        })
        modules.append({
            "type" : "heatmap",
            "stats" : False,
            "window" : "1000",
            "linear" : True,
            "layers" : [
                {"layer" : "sc_sup" },
                {"layer" : "sc_deep" },
                {"layer" : "gating" },
            ]
        })

    return modules

def build_bridge_connections():
    return [
        {
            "from structure" : "nvm",
            "from layer" : "fef",
            "to structure" : "oculomotor",
            "to layer" : "sc_deep",
            "type" : "divergent",
            "convolutional" : True,
            "opcode" : "add",
            "plastic" : False,
            "weight config" : {
                "type" : "flat",
                "weight" : 0.2,
                "from spacing" : 4,
                "distance callback" : "gaussian",
            },
            "arborized config" : {
                "field size" : 5,
                "stride" : 4,
                "wrap" : False
            }
        },
        {
            "from structure" : "nvm",
            "from layer" : "fef",
            "to structure" : "oculomotor",
            "to layer" : "gating",
            "dendrite" : "disinhibition",
            "type" : "divergent",
            "convolutional" : True,
            "opcode" : "sub",
            "plastic" : False,
            "weight config" : {
                "type" : "flat",
                "weight" : 4.0,
                "from spacing" : 4,
                "distance callback" : "gaussian",
            },
            "arborized config" : {
                "field size" : 5,
                "stride" : 4,
                "wrap" : False
            }
        },
        {
            "from structure" : "nvm",
            "from layer" : "sc",
            "to structure" : "oculomotor",
            "to layer" : "gating",
            "dendrite" : "disinhibition",
            "type" : "fully connected",
            "opcode" : "mult",
            "plastic" : False,
            "weight config" : {
                "type" : "flat",
                "weight" : 1.0 / 25,
                "distance callback" : "gaussian",
            }
        }
    ]

def main(read=True, visualizer=False, device=None, rate=0, iterations=1000000):
    ''' Build oculomotor model '''
    task = "dsst"
    task = "saccade"

    # Scale for superior colliculus
    scale = 5

    if task == "dsst":
        rows = dsst_params["input rows"]
        cols = dsst_params["input columns"]
    elif task == "saccade":
        rows = 340
        cols = 480
    else:
        raise ValueError

    om_network = build_network(rows, cols, scale)
    om_modules = build_environment(rows, cols, scale, visualizer, task)

    ''' Build NVM '''
    np.set_printoptions(linewidth=200, formatter = {'float': lambda x: '% .2f'%x})

    #nvmnet = make_saccade_nvm("tanh")
    nvmnet = make_saccade_nvm("logistic")

    print(nvmnet.layers["gh"].activator.off)
    print(nvmnet.w_gain, nvmnet.b_gain)
    print(nvmnet.layers["go"].activator.label)
    print(nvmnet.layers["gh"].activator.label)
    raw_input("continue?")

    if visualizer:
        viz_layers = ["sc","fef","tc","ip","opc","op1","op2","gh","go"]
    else:
        viz_layers = []

    nvm_structure, nvm_connections = make_syngen_network(nvmnet)
    nvm_modules = make_syngen_environment(nvmnet,
        initial_patterns = dict(nvmnet.activity),
        run_nvm=False,
        viz_layers = viz_layers,
        #print_layers = nvmnet.layers,
        # stat_layers=["ip","go","gh"])
        stat_layers=[],
        read=read)

    bridge_connections = build_bridge_connections()

    net = Network({"structures" : [nvm_structure] + om_network["structures"],
                   "connections" : nvm_connections + om_network["connections"] + bridge_connections})
    env = Environment({"modules" : nvm_modules + om_modules})

    init_syngen_nvm(nvmnet, net)


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
                               "learning flag" : False})
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
