import sys
sys.path.append('../nvm')

from gate_sequencer import GateSequencer

import numpy as np
from syngen import Network, Environment, create_io_callback, FloatArray
from syngen import get_cpu, get_gpus, interrupt_engine

class SyngenNVM:
    def __init__(self, nvmnet, auxiliary=[]):
        self.auxiliary = auxiliary
        aux_structures = [aux.structure for aux in auxiliary]
        aux_conns = [c for aux in auxiliary for c in aux.connections]

        structure, connections = make_syngen_network(nvmnet)
        self.net = Network({
            "structures" : [structure] + aux_structures,
            "connections" : connections + aux_conns})

        # Initialize bias
        self.get_output("bias")[0] = 1.0

        self.initialize_weights(nvmnet)
        self.initialize_activity(nvmnet)

    def initialize_weights(self, nvmnet):
        init_syngen_nvm_weights(nvmnet, self.net)
        for aux in self.auxiliary:
            aux.initialize_weights(self, nvmnet)

    def initialize_activity(self, nvmnet):
        try: init_syngen_nvm_activity(nvmnet.activity, self.net)
        except AttributeError: pass
        for aux in self.auxiliary:
            aux.initialize_activity(self, nvmnet)

    def get_output(self, layer_name):
        return self.net.get_neuron_data(
            "nvm", layer_name, "output").to_np_array()

    def decode_output(self, layer_name, coder):
        return coder.decode(self.get_output(layer_name))

    def run(self, syn_env=None, args={}):
        modules = [] if syn_env is None else syn_env.modules

        default_args = {
            "multithreaded" : False,
            "worker threads" : 0,
            "iterations" : 0,
            "refresh rate" : 0,
            "verbose" : False,
            "learning flag" : False}

        for key,val in default_args.items():
            if key not in args:
                args[key] = val

        env = Environment({"modules" : modules})
        report = self.net.run(env, args)
        env.free()
        return report

    def free(self):
        self.net.free()
        self.net = None

class SyngenEnvironment:
    def __init__(self):
        self.modules = [make_exit_module()]

    def add_visualizer(self, structure, layer_names):
        self.modules.append(make_visualizer_module(structure, layer_names))

    def add_printer(self, nvmnet, layer_names):
        self.modules.append(make_printer_module(nvmnet, layer_names))

    def add_checker(self, nvmnet):
        self.modules.append(make_checker_module(nvmnet))

    def add_custom_input(self, structure, layer_names, cb_name, cb, clear=True):
        self.modules.append(
            make_custom_input_module(structure, layer_names, cb_name, cb, clear))

    def add_custom_output(self, structure, layer_names, cb_name, cb):
        self.modules.append(
            make_custom_output_module(structure, layer_names, cb_name, cb))

    def add_streams(self, nvmnet, op_reg, data_reg, producer, consumer):
        self.modules += make_stream_modules(
            nvmnet, op_reg, data_reg, producer, consumer)



# Builds a name for a connection
def get_conn_name(to_name, from_name, suffix=""):
    return "%s<%s%s" % (
        to_name, from_name, "" if suffix is None else ("-%s" % suffix))

def make_syngen_network(nvmnet):
    for layer in nvmnet.layers.values():
        if layer.activator.label not in ["tanh", "heaviside"]:
            raise ValueError("Syngen NVM must use tanh/heaviside")

    ### LAYERS ###
    layer_configs = []

    for layer_name, layer in nvmnet.layers.items():
        # 'go' and 'co' use special neural models
        if layer_name == "go":
            model = "nvm_heaviside"
        elif layer_name == "co":
            model = "nvm_compare"
        else:
            model = "nvm"

        # Build layer config
        layer_configs.append({
            "name" : layer_name,
            "neural model" : model,
            "rows" : layer.shape[0],
            "columns" : layer.shape[1],
        })

    # bias layer
    layer_configs.append({
        "name" : "bias",
        "neural model" : "relay",
        "rows" : 1,
        "columns" : 1,
        "init config": {
            "type": "flat",
            "value": 1
        }
    })

    # exit detection layer
    opc = nvmnet.layers["opc"]
    exit_bias = opc.activator.on * -(opc.size - 1)
    layer_configs.append({
        "name" : "exit",
        "neural model" : "relay",
        "rows" : 1,
        "columns" : 1,
        "init config": {
            "type": "flat",
            "value": exit_bias
        }
    })

    structure = {"name" : "nvm",
                 "type" : "parallel",
                 "layers": layer_configs}


    ### CONNECTIONS ###
    connections = []

    # Exit detection connection
    connections.append({
        "name" : get_conn_name("exit", "opc"),
        "from layer" : "opc",
        "to layer" : "exit",
        "type" : "fully connected",
        "plastic" : False,
    })

    # Determine which gates are always active to save computations
    #   For decay gates, omit gain connection
    #   For update gates, use non-gated kernel
    gate_map, layers, registers = nvmnet.gate_map, nvmnet.layers, nvmnet.registers
    gate_output, gate_hidden = layers['go'], layers['gh']
    gs = GateSequencer(gate_map, gate_output, gate_hidden, layers)
    gates_always_on = np.where(gs.make_gate_output() > 0.0)[0]

    # Add gain connections (skip if decay always on)
    for layer in nvmnet.layers.values():
        decay_gate_index = nvmnet.gate_map.get_gate_index(
            (layer.name, layer.name, "d"))

        if decay_gate_index not in gates_always_on:
            connections.append({
                "name" : get_conn_name(layer.name, layer.name, "decay"),
                "from layer" : "go",
                "to layer" : layer.name,
                "type" : "subset",
                "subset config" : {
                    "from row end" : 1,
                    "from column start" : decay_gate_index,
                    "from column end" : decay_gate_index+1,
                    "to row end" : 1,
                    "to column end" : 1,
                },
                "plastic" : False,
                "decay" : True
            })

            connections.append({
                "name" : get_conn_name(layer.name, layer.name, "gain"),
                "from layer" : layer.name,
                "to layer" : layer.name,
                "type" : "one to one",
                "opcode" : "add",
                "weight config" : {
                    "type" : "flat",
                    "weight" : nvmnet.w_gain[layer.name]
                },
                "plastic" : False,
                "gated" : True
            })

    # Build standard connections and their gates
    for (to_name, from_name) in nvmnet.weights:
        to_layer, from_layer = nvmnet.layers[to_name], nvmnet.layers[from_name]

        # Activity gate (skip if always on)
        gate_index = nvmnet.gate_map.get_gate_index(
                (to_name, from_name, "u"))
        gated = gate_index not in gates_always_on
        if gated:
            connections.append({
                "name" : get_conn_name(to_name, from_name, "gate"),
                "from layer" : "go",
                "to layer" : to_name,
                "type" : "subset",
                "subset config" : {
                    "from row end" : 1,
                    "from column start" : gate_index,
                    "from column end" : gate_index+1,
                    "to row end" : 1,
                    "to column end" : 1,
                },
                "plastic" : False,
                "gate" : True
            })

        # Connections with gated learning
        #   register <- mf
        #   co <- ci
        #   ip <- sf
        #   mf <- register
        #   mb <- register
        #   mp <- mf
        plastic = (to_name in nvmnet.registers and from_name == "mf") or \
           (to_name == "co") or \
           (to_name == "ip" and from_name == "sf") or \
           (to_name in ["mf", "mb"] and from_name in nvmnet.registers) or \
           (to_name == 'mp' and from_name == 'mf')

        # Learning gate (skip if never used)
        if plastic:
            gate_index = nvmnet.gate_map.get_gate_index(
                (to_name, from_name, "l"))
            connections.append({
                "name" : get_conn_name(to_name, from_name, "learning"),
                "from layer" : "go",
                "to layer" : to_name,
                "type" : "subset",
                "subset config" : {
                    "from row end" : 1,
                    "from column start" : gate_index,
                    "from column end" : gate_index+1,
                    "to row end" : 1,
                    "to column end" : 1,
                },
                "plastic" : False,
                "learning" : True
            })

        # Normalization factor for plasticity
        if to_name == "co":
            norm = to_layer.activator.on ** 2
        else:
            norm = from_layer.size * (to_layer.activator.on ** 2)

        # Weights
        connections.append({
            "name" : get_conn_name(to_name, from_name, "weights"),
            "from layer" : from_name,
            "to layer" : to_name,
            "type" : "fully connected",
            "opcode" : "add",
            "gated" : gated,
            "plastic" : plastic,
            "norm" : norm,
        })

        # Biases (skip if all zero)
        if np.count_nonzero(nvmnet.biases[(to_name, from_name)]) > 0:
            connections.append({
                "name" : get_conn_name(to_name, from_name, "biases"),
                "from layer" : "bias",
                "to layer" : to_name,
                "type" : "fully connected",
                "opcode" : "add",
                "gated" : gated,
                "plastic" : False,
            })

    # Set structures
    for conn in connections:
        conn["from structure"] = "nvm"
        conn["to structure"] = "nvm"

    return structure, connections


def init_syngen_nvm_weights(nvmnet, syngen_net):
    # Initialize weights
    for (to_name, from_name), w in nvmnet.weights.items():
        if np.count_nonzero(w) > 0:
            syngen_net.get_weight_matrix(
                get_conn_name(to_name, from_name, "weights")).copy_from(w.flat)

    # Initialize biases
    for (to_name, from_name), b in nvmnet.biases.items():
        if np.count_nonzero(b) > 0:
            syngen_net.get_weight_matrix(
                get_conn_name(to_name, from_name, "biases")).copy_from(b.flat)

    # Initialize exit detector
    exit_pattern = np.sign(nvmnet.layers["opc"].coder.encode("exit"))
    syngen_net.get_weight_matrix(
        get_conn_name("exit", "opc")).copy_from(exit_pattern.flat)

    # Initialize comparison true pattern
    co = nvmnet.layers["co"]
    co_true = co.coder.encode("true")
    syngen_net.get_neuron_data(
        "nvm", "co", "true_state").copy_from(co.activator.g(co_true.flat))

def init_syngen_nvm_activity(activity, syngen_net):
    for layer_name, activity in activity.items():
        if np.count_nonzero(activity) > 0:
            syngen_net.get_neuron_data(
                "nvm", layer_name, "output").copy_from(activity.flat)


def make_visualizer_module(structure, layer_names):
    return {
        "type" : "visualizer",
        "layers" : [
            {"structure": structure, "layer": layer_name}
                for layer_name in layer_names]
    }

def make_exit_module():

    def exit_callback(ID, size, ptr):
        if FloatArray(size,ptr)[0] > 0.0:
            interrupt_engine()

    create_io_callback("nvm_exit", exit_callback)

    return {
        "type" : "callback",
        "layers" : [
            {
                "structure" : "nvm",
                "layer" : "exit",
                "output" : True,
                "function" : "nvm_exit",
                "id" : 0
            }
        ]
    }


def make_printer_module(nvmnet, layer_names):

    first = layer_names[0] if len(layer_names) > 0 else ""

    def print_callback(ID, size, ptr):
        layer_name = layer_names[ID]
        if layer_name == first:
            print("")
        syn_v = FloatArray(size,ptr).to_np_array()
        syn_tok = nvmnet.layers[layer_name].coder.decode(syn_v)
        print("%4s: %12s"%(layer_name, syn_tok))

    create_io_callback("nvm_print", print_callback)

    return {
        "type" : "callback",
        "layers" : [
            {
                "structure" : "nvm",
                "layer" : layer_name,
                "output" : True,
                "function" : "nvm_print",
                "id" : i
            } for i,layer_name in enumerate(layer_names)
        ]
    }

def make_checker_module(nvmnet):

    layer_names = list(nvmnet.layers.keys())

    def checker_callback(ID, size, ptr):
        if ID == 0:
            nvmnet.tick()

        layer_name = layer_names[ID]

        coder = nvmnet.layers[layer_name].coder
        syn_v = FloatArray(size,ptr).to_np_array()
        syn_tok = coder.decode(syn_v)
        py_v = nvmnet.activity[layer_name]
        py_tok = coder.decode(py_v)

        if py_tok != syn_tok:
            residual = np.fabs(syn_v.reshape(py_v.shape) - py_v).max()

            print("Mismatch detected in nvm_checker!")
            print("%4s: %12s | %12s (res=%f)"%(
                layer_name, syn_tok, py_tok, residual))

            interrupt_engine()

    create_io_callback("nvm_checker", checker_callback)

    return {
        "type" : "callback",
        "layers" : [
            {
                "structure" : "nvm",
                "layer" : layer_name,
                "output" : True,
                "function" : "nvm_checker",
                "id" : i
            } for i,layer_name in enumerate(layer_names)
        ]
    }

def make_custom_input_module(structure, layer_names, name, cb, clear):

    def custom_callback(ID, size, ptr):
        cb(layer_names[ID], FloatArray(size,ptr).to_np_array())

    create_io_callback(name, custom_callback)

    return {
        "type" : "callback",
        "clear" : clear,
        "layers" : [
            {
                "structure" : structure,
                "layer" : layer_name,
                "input" : True,
                "function" : name,
                "id" : i
            } for i,layer_name in enumerate(layer_names)
        ]
    }

def make_custom_output_module(structure, layer_names, name, cb):

    def custom_callback(ID, size, ptr):
        cb(layer_names[ID], FloatArray(size,ptr).to_np_array())

    create_io_callback(name, custom_callback)

    return {
        "type" : "callback",
        "layers" : [
            {
                "structure" : structure,
                "layer" : layer_name,
                "output" : True,
                "function" : name,
                "id" : i
            } for i,layer_name in enumerate(layer_names)
        ]
    }

def make_stream_modules(nvmnet, op_reg, data_reg, producer, consumer):
    data_layer = nvmnet.layers[data_reg]
    op_layer = nvmnet.layers[op_reg]
    read_pattern = nvmnet.w_gain[op_reg] * op_layer.coder.encode("read")
    write_pattern = nvmnet.w_gain[op_reg] * op_layer.coder.encode("write")
    null_pattern = nvmnet.w_gain[op_reg] * op_layer.coder.encode("null")

    class StreamState:
        # waiting, read, write, reset
        state = "waiting"

    def stream_data_in(name, data):
        if StreamState.state == "read":
            ready,sym = producer()
            if ready:
                StreamState.state = "reset"
                data += data_layer.activator.g(data_layer.coder.encode(sym)).flat
                data *= 10

    def stream_data_out(name, data):
        if StreamState.state == "write":
            ready = consumer(nvmnet.layers[name].coder.decode(data))
            if ready:
                StreamState.state = "reset"

    def stream_operator_in(name, data):
        if StreamState.state == "reset":
            data += null_pattern.flat
            data *= 10
            StreamState.state = "waiting"

    def stream_operator_out(name, data):
        output = nvmnet.layers[name].coder.decode(data)

        if output in ["read", "write"]:
            StreamState.state = output

    modules = []

    modules.append(make_custom_input_module("nvm", [data_reg],
        "stream_data_in", stream_data_in, clear=True))
    modules.append(make_custom_input_module("nvm", [op_reg],
        "stream_operator_in", stream_operator_in, clear=True))

    modules.append(make_custom_output_module("nvm", [op_reg],
        "stream_operator_out", stream_operator_out))
    modules.append(make_custom_output_module("nvm", [data_reg],
        "stream_data_out", stream_data_out))

    return modules
