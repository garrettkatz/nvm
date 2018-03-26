from syngen import Network, Environment
from syngen import get_gpus, get_cpu
from syngen import set_suppress_output, set_warnings, set_debug
from syngen import create_distance_weight_callback, FloatArray

from os import path
from math import exp
import sys
import argparse

import numpy as np
from saccade_programs import make_saccade_nvm
from nvm_to_syngen import make_syngen_network, make_syngen_environment, init_syngen_nvm
from syngen import ConnectionFactory, create_io_callback, FloatArray

def get_dsst_params(num_rows=8, cell_res=8):
    num_cols = 18
    cell_rows = 2*cell_res + 1
    spacing = cell_res / 4

    input_rows = (num_rows + 2) * (cell_rows + spacing) - spacing
    input_cols = num_cols * (cell_res + spacing) - spacing

    focus_rows = input_rows - cell_rows
    focus_cols = input_cols - cell_res

    return {
        "columns" : num_cols,
        "rows" : num_rows,
        "cell columns" : cell_res,
        "cell rows" : cell_rows,
        "spacing" : spacing,
        "input rows" : input_rows,
        "input columns" : input_cols,
        "focus rows" : focus_rows,
        "focus columns" : focus_cols,
    }

dsst_params = get_dsst_params(num_rows=8, cell_res=8)

def gauss(dist, peak, sig, norm=False):
    inv_sqrt_2pi = 0.3989422804014327
    peak_coeff = peak * (1.0 if norm else inv_sqrt_2pi / sig)

    a = dist / sig
    return peak_coeff * exp(-0.5 * a * a)

# Create distance weight init callback
def dist_callback(ID, size, weights, distances):
    w_arr = FloatArray(size, weights)
    d_arr = FloatArray(size, distances)

    # Find maximum distance and half it
    max_dist = max(d_arr.data[i] for i in xrange(size)) / 2

    # Smooth out weights based on distances
    for i in xrange(size):
        w_arr.data[i] = gauss(d_arr.data[i], w_arr.data[i], max_dist, True)

create_distance_weight_callback("gaussian", dist_callback)

def build_exc_inh_pair(
        exc_name,
        inh_name,
        rows = 200,
        cols = 200,

        half_inh = True,
        mask = True,

        exc_tau = 0.2,
        inh_tau = 0.2,

        exc_decay = 0.1,
        inh_decay = 0.1,

        exc_noise_rate = 1,
        inh_noise_rate = 1,
        exc_noise_random = "false",
        inh_noise_random = "false",

        exc_exc_rf = 31,
        exc_inh_rf = 123,
        inh_exc_rf = 83,
        inh_inh_rf = 63,

        mask_rf = 31,

        exc_exc_fraction = 1,
        exc_inh_fraction = 1,
        inh_exc_fraction = 1,
        inh_inh_fraction = 1,

        exc_exc_mean = 0.05,
        exc_inh_mean = 0.025,
        inh_exc_mean = 0.025,
        inh_inh_mean = 0.025,

        exc_exc_std_dev = 0.01,
        exc_inh_std_dev = 0.005,
        inh_exc_std_dev = 0.005,
        inh_inh_std_dev = 0.005):

    exc_noise_strength = 0.5 / exc_tau
    inh_noise_strength = 0.5 / inh_tau
    exc = {
        "name" : exc_name,
        "neural model" : "oscillator",
        "rows" : rows,
        "columns" : cols,
        "tau" : exc_tau,
        "decay" : exc_decay,
        "init config" : {
            "type" : "poisson",
            "value" : exc_noise_strength,
            "rate" : exc_noise_rate,
            "random" : exc_noise_random
        }}
    inh = {
        "name" : inh_name,
        "neural model" : "oscillator",
        "rows" : rows/2 if half_inh else rows,
        "columns" : cols/2 if half_inh else cols,
        "tau" : inh_tau,
        "decay" : inh_decay,
        "init config" : {
            "type" : "poisson",
            "value" : inh_noise_strength,
            "rate" : inh_noise_rate,
            "random" : inh_noise_random
        }}

    connections = [
        {
            "from layer" : exc_name,
            "to layer" : exc_name,
            "type" : "convergent",
            "convolutional" : "true",
            "opcode" : "add",
            "plastic" : "false",
            "weight config" : {
                "type" : "gaussian",
                "mean" : exc_exc_mean,
                "std dev" : exc_exc_std_dev,
                "fraction" : exc_exc_fraction,
                "diagonal" : "false",
                "circular mask" : [ { } ],
                "distance callback" : "gaussian",
            },
            "arborized config" : {
                "field size" : exc_exc_rf,
                "stride" : 1,
                "wrap" : "false"
            }
        },
        {
            "from layer" : exc_name,
            "to layer" : inh_name,
            "type" : "convergent",
            "convolutional" : "true",
            "opcode" : "add",
            "plastic" : "false",
            "weight config" : {
                "type" : "gaussian",
                "mean" : exc_inh_mean,
                "std dev" : exc_inh_std_dev,
                "fraction" : exc_inh_fraction,
                "circular mask" : [
                    {
                        "diameter" : mask_rf,
                        "invert" : "true"
                    },
                    { }
                ] if mask else [ { } ],
                "distance callback" : "gaussian",
                "to spacing" : 2,
            },
            "arborized config" : {
                "field size" : exc_inh_rf,
                "stride" : 2 if half_inh else 1,
                "wrap" : "true"
            }
        },
        {
            "from layer" : inh_name,
            "to layer" : exc_name,
            "type" : "divergent" if half_inh else "convergent",
            "convolutional" : "true",
            "opcode" : "sub",
            "plastic" : "false",
            "weight config" : {
                "type" : "gaussian",
                "mean" : inh_exc_mean,
                "std dev" : inh_exc_std_dev,
                "fraction" : inh_exc_fraction,
                "circular mask" : [ { } ] if not half_inh else None,
                "distance callback" : "gaussian",
                "from spacing" : 2,
            },
            "arborized config" : {
                "field size" : inh_exc_rf,
                "stride" : 2 if half_inh else 1,
                "wrap" : "true"
            }
        },
#        {
#            "from layer" : inh_name,
#            "to layer" : inh_name,
#            "type" : "convergent",
#            "convolutional" : "true",
#            "opcode" : "sub",
#            "plastic" : "false",
#                "weight config" : {
#                "type" : "gaussian",
#                "mean" : inh_inh_mean,
#                "std dev" : inh_inh_std_dev,
#                "fraction" : inh_inh_fraction,
#                "circular mask" : [ { } ]
#            },
#            "arborized config" : {
#                "field size" : inh_inh_rf,
#                "stride" : 1,
#                "wrap" : "false"
#            }
#        },
    ]

    return [exc, inh], connections


def build_network(rows=200, cols=200, scale=5):
    dim = min(rows, cols)

    #sc_rf_scales = (7, 2, 2.5, 3.5)
    #motor_rf_scales = (2, 2.5, 3.0)

    sc_rf_scales = (30, 5, 5, 15)
    motor_rf_scales = (2, 2.5, 3.0)

    sc_rf_scales = (60, 10, 10, 30)
    motor_rf_scales = (10, 2, 2, 50)

    # Create main structure (parallel engine)
    structure = {"name" : "oculomotor", "type" : "parallel"}

    # Add retinal layer
    vision_layer = {
        "name" : "vision",
        "neural model" : "oscillator",
        "rows" : rows,
        "columns" : cols}

    sc_layers, sc_conns = build_exc_inh_pair(
        "sc_exc", "sc_inh",
        rows, cols,
        half_inh = True,
        mask = True,

        exc_tau = 0.1,
        inh_tau = 0.1,

        exc_decay = 0.05,
        inh_decay = 0.05,

        exc_noise_rate = 0,
        inh_noise_rate = 0,
        exc_noise_random = "false",
        inh_noise_random = "false",

        exc_exc_rf = dim/sc_rf_scales[0],
        exc_inh_rf = dim/sc_rf_scales[1],
        inh_exc_rf = dim/sc_rf_scales[2],
        inh_inh_rf = 1, #dim/sc_rf_scales[3],

        mask_rf = dim/sc_rf_scales[3],

        exc_exc_fraction = 0.25,
        exc_inh_fraction = 1,
        inh_exc_fraction = 1,
        inh_inh_fraction = 1,

        exc_exc_mean = 0.05,
        exc_inh_mean = 0.025,
        inh_exc_mean = 0.025,
        inh_inh_mean = 0.025,

        exc_exc_std_dev = 0.01,
        exc_inh_std_dev = 0.005,
        inh_exc_std_dev = 0.005,
        inh_inh_std_dev = 0.005)

    motor_rows = int(rows/scale)
    motor_cols = int(cols/scale)
    motor_dim = min(motor_rows, motor_cols)
    sc_out_layers, sc_out_conns = build_exc_inh_pair(
        "sc_out_exc", "sc_out_inh",
        motor_rows, motor_cols,
        half_inh = True,
        mask = True,

        exc_tau = 0.4,
        inh_tau = 0.8,

        exc_decay = 0.1,
        inh_decay = 0.05,

        exc_noise_rate = 0,
        inh_noise_rate = 0,
        exc_noise_random = "false",
        inh_noise_random = "false",

        exc_exc_rf = motor_dim/motor_rf_scales[0],
        exc_inh_rf = motor_dim/motor_rf_scales[1],
        inh_exc_rf = motor_dim/motor_rf_scales[2],
        inh_inh_rf = 1, #motor_dim/motor_rf_scales[3],

        mask_rf = motor_dim/motor_rf_scales[3],

        exc_exc_fraction = 1,
        exc_inh_fraction = 1,
        inh_exc_fraction = 1,
        inh_inh_fraction = 1,

        exc_exc_mean = 0.1,
        exc_inh_mean = 0.05,
        inh_exc_mean = 0.05,
        inh_inh_mean = 0.05,

        exc_exc_std_dev = 0.02,
        exc_inh_std_dev = 0.01,
        inh_exc_std_dev = 0.01,
        inh_inh_std_dev = 0.01)

    '''
    gating_layer = {
        "name" : "gating",
        "neural model" : "oscillator",
        "rows" : motor_rows,
        "columns" : motor_cols,
        "tau" : 0.05,
        "decay" : 0.05,
        "tonic" : 0.0}
    '''

    # Add layers to structure
    structure["layers"] = [vision_layer] + \
        sc_layers + sc_out_layers
        #sc_layers + sc_out_layers + [gating_layer]

    # Create connections
    receptive_field = 31
    connections = [
        {
            "from layer" : "vision",
            "to layer" : "sc_exc",
            "type" : "convergent",
            "convolutional" : "true",
            "opcode" : "add",
            "plastic" : "false",
            "weight config" : {
                "type" : "flat",
                "weight" : 0.01,
                "distance callback" : "gaussian",
            },
            "arborized config" : {
                "field size" : receptive_field,
                "stride" : 1,
                "wrap" : "false"
            }
        },
        {
            "from layer" : "sc_exc",
            "to layer" : "sc_out_exc",
            "type" : "convergent",
            "convolutional" : "true",
            "opcode" : "add",
            "plastic" : "false",
            "weight config" : {
                "type" : "flat",
                "weight" : 0.5,
            },
            "arborized config" : {
                "field size" : rows/motor_rows,
                "stride" : cols/motor_cols,
                "wrap" : "false",
                "offset" : 0,
                "distance callback" : "gaussian",
                "to spacing" : cols / motor_cols
            }
        },
#        {
#            "from layer" : "gating",
#            "to layer" : "sc_out_exc",
#            "type" : "one to one",
#            "opcode" : "mult",
#            "plastic" : "false",
#            "weight config" : {
#                "type" : "flat",
#                "weight" : 1.0,
#            }
#        }
        ] + sc_conns + sc_out_conns

    # Set structures
    for conn in connections:
        conn["from structure"] = "oculomotor"
        conn["to structure"] = "oculomotor"

    return structure, connections

def build_environment(rows=200, cols=200, scale=5, visualizer=False, dsst=False, saccade=True):
    dim = min(rows, cols)
    motor_dim = min(rows/scale, cols/scale)

    if dsst == saccade:
        raise ValueError

    # Create environment modules
    modules = [
        {
            "type" : "saccade",
            "layers" : [
                {
                    "structure" : "oculomotor",
                    "layer" : "vision",
                    "params" : "input",
                },
                {
                    "structure" : "oculomotor",
                    "layer" : "sc_out_exc",
                    "output" : "true",
                }
            ]
        } if saccade else {
            "type" : "dsst",
            "rows" : dsst_params["rows"],
            "columns" : dsst_params["columns"],
            "cell size" : dsst_params["cell columns"],
            "layers" : [
                {
                    "structure" : "oculomotor",
                    "layer" : "vision",
                    "input" : "true",
                }
            ]
        },
#        {
#            "type" : "gaussian_random_input",
#            "rate" : "100",
#            "border" : motor_dim/5,
#            "std dev" : motor_dim/10,
#            "value" : 5.0,
#            "normalize" : "true",
#            "peaks" : "1",
#            "random" : "false",
#            "layers" : [
#                {
#                    "structure" : "oculomotor",
#                    "layer" : "gating"
#                }
#            ]
#        }
    ]
    if visualizer:
        modules.append({
            "type" : "visualizer",
            "layers" : [
#                { "structure" : "oculomotor", "layer" : "vision" },
                { "structure" : "oculomotor", "layer" : "sc_exc" },
                { "structure" : "oculomotor", "layer" : "sc_inh" },
                { "structure" : "oculomotor", "layer" : "sc_out_exc" },
                { "structure" : "oculomotor", "layer" : "sc_out_inh" },
#                { "structure" : "oculomotor", "layer" : "gating" },
            ]
        })
        modules.append({
            "type" : "heatmap",
            "stats" : "false",
            "window" : "1000",
            "linear" : "true",
            "layers" : [
#                { "structure" : "oculomotor", "layer" : "vision" },
                { "structure" : "oculomotor", "layer" : "sc_exc" },
                { "structure" : "oculomotor", "layer" : "sc_inh" },
                { "structure" : "oculomotor", "layer" : "sc_out_exc" },
                { "structure" : "oculomotor", "layer" : "sc_out_inh" },
#                { "structure" : "oculomotor", "layer" : "gating" },
            ]
        })

    return modules

def build_bridge_connections():
    return [
        {
            "from structure" : "nvm",
            "from layer" : "fef",
            "to structure" : "oculomotor",
            "to layer" : "sc_exc",
            "type" : "divergent",
            "convolutional" : "true",
            "opcode" : "add",
            "plastic" : "false",
            "weight config" : {
                "type" : "flat",
                "weight" : 0.5,
            },
            "arborized config" : {
                "field size" : 5,
                "stride" : 5,
                "wrap" : "false"
            }
        },
        {
            "from structure" : "nvm",
            "from layer" : "sc",
            "to structure" : "oculomotor",
            "to layer" : "sc_out_exc",
            "type" : "fully connected",
            "opcode" : "mult",
            "plastic" : "false",
            "weight config" : {
                "type" : "flat",
                "weight" : 0.5,
            }
        }
    ]

def main(visualizer=False, device=None, rate=0, iterations=1000000):
    ''' Build oculomotor model '''
    dsst = False
    saccade = True

    #rows = 100
    #cols = 200
    scale = 5

    if dsst:
        rows = dsst_params["input rows"]
        cols = dsst_params["input columns"]
    if saccade:
        rows = 340
        cols = 480

    om_structure, om_connections = build_network(rows, cols, scale)
    om_modules = build_environment(rows, cols, scale, visualizer, dsst, saccade)

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
        print_layers = nvmnet.layers,
        # stat_layers=["ip","go","gh"])
        stat_layers=[])

    bridge_connections = build_bridge_connections()

    net = Network({"structures" : [nvm_structure, om_structure],
                   "connections" : nvm_connections + om_connections + bridge_connections})
    env = Environment({"modules" : nvm_modules + om_modules})

    init_syngen_nvm(nvmnet, net)


    ''' Run Simulation '''
    if device is None:
        device = gpus[len(gpus)-1] if len(gpus) > 0 else get_cpu()

    report = net.run(env, {"multithreaded" : "true",
                               "worker threads" : "4",
                               "devices" : device,
                               "iterations" : iterations,
                               "refresh rate" : rate,
                               "verbose" : "true",
                               "learning flag" : "false"})
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

    main(args.visualizer, device, args.r)
