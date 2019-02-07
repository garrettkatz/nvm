import numpy as np
from syngen import Network, Environment, ConnectionFactory, create_io_callback, FloatArray, VoidArray, interrupt_engine

class SyngenNVM:
    def __init__(self, nvmnet):
        pass

# Builds a name for a connection or dendrite
def build_name(from_name, to_name, suffix=""):
    return to_name + "<" + from_name + suffix

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

    # one more bias layer
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

    structure = {"name" : "nvm",
                 "type" : "parallel",
                 "layers": layer_configs}


    ### CONNECTIONS ###
    connections = []

    # Add gain connections
    gate_output = nvmnet.layers["go"]
    for layer in nvmnet.layers.values():
        # Gate layer decay always active
        if layer not in ["go", "gh"]:
            decay_gate_index = nvmnet.gate_map.get_gate_index(
                (layer.name, layer.name, "d"))
            connections.append({
                "name" : "%s-decay" % layer.name,
                "from layer" : "go",
                "to layer" : layer.name,
                "type" : "subset",
                "subset config" : {
                    "from row start" : 0,
                    "from row end" : 1,
                    "from column start" : decay_gate_index,
                    "from column end" : decay_gate_index+1,
                    "to row start" : 0,
                    "to row end" : 1,
                    "to column start" : 0,
                    "to column end" : 1,
                },
                "plastic" : False,
                "decay" : True
            })

            connections.append({
                "name" : "%s-gain" % layer.name,
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

        # Gate mechanism is always running
        gated = (from_name != "gh")

        if gated:
            # Activity gate
            gate_index = nvmnet.gate_map.get_gate_index(
                (to_layer.name, from_layer.name, "u"))
            connections.append({
                "name" : build_name(from_layer.name, to_layer.name, "-gate"),
                "from layer" : "go",
                "to layer" : to_layer.name,
                "type" : "subset",
                "subset config" : {
                    "from row start" : 0,
                    "from row end" : 1,
                    "from column start" : gate_index,
                    "from column end" : gate_index+1,
                    "to row start" : 0,
                    "to row end" : 1,
                    "to column start" : 0,
                    "to column end" : 1,
                },
                "plastic" : False,
                "gate" : True
            })

        # Plastic connections:
        #   device <- mf
        #   co <- ci
        #   ip <- sf
        #   mf <- device
        #   mb <- device
        plastic = (to_name in nvmnet.devices and from_name == "mf") or \
           (to_name == "co") or \
           (to_name == "ip" and from_name == "sf") or \
           (to_name in ["mf", "mb"] and from_name in nvmnet.devices)

        # Normalization factor for plasticity
        if to_name == "co":
            norm = to_layer.activator.on ** 2
        else:
            norm = from_layer.size * (to_layer.activator.on ** 2)

        if plastic:
            # Learning gate
            gate_index = nvmnet.gate_map.get_gate_index(
                (to_layer.name, from_layer.name, "l"))
            connections.append({
                "name" : build_name(from_layer.name, to_layer.name, "-learning"),
                "from layer" : "go",
                "to layer" : to_layer.name,
                "type" : "subset",
                "subset config" : {
                    "from row start" : 0,
                    "from row end" : 1,
                    "from column start" : gate_index,
                    "from column end" : gate_index+1,
                    "to row start" : 0,
                    "to row end" : 1,
                    "to column start" : 0,
                    "to column end" : 1,
                },
                "plastic" : False,
                "learning" : True
            })

        # Weights
        connections.append({
            "name" : build_name(from_layer.name, to_layer.name, "-weights"),
            "from layer" : from_layer.name,
            "to layer" : to_layer.name,
            "type" : "fully connected",
            "opcode" : "add",
            "weight config" : {
                "type" : "flat",
                "weight" : 0.0  # This will get overwritten when the ROM is flashed
            },
            "gated" : gated,
            "plastic" : plastic,
            "norm" : norm,
        })

        # Biases
        connections.append({
            "name" : build_name(from_layer.name, to_layer.name, "-biases"),
            "from layer" : "bias",
            "to layer" : to_layer.name,
            "type" : "fully connected",
            "opcode" : "add",
            "weight config" : {
                "type" : "flat",
                "weight" : 0.0  # This will get overwritten when the ROM is flashed
            },
            "gated" : gated,
            "plastic" : False,
        })

    # Set structures
    for conn in connections:
        conn["from structure"] = "nvm"
        conn["to structure"] = "nvm"

    return structure, connections


def init_syngen_nvm(nvmnet, syngen_net):
    # Initialize weights
    for (to_name, from_name), w in nvmnet.weights.items():
        mat = syngen_net.get_weight_matrix(
            build_name(from_name, to_name, "-weights"))
        if not isinstance(mat, VoidArray):
            for m in range(mat.size):
                mat.data[m] = w.flat[m]

    # Initialize biases
    for (to_name, from_name), b in nvmnet.biases.items():
        mat = syngen_net.get_weight_matrix(
            build_name(from_name, to_name, "-biases"))
        if not isinstance(mat, VoidArray):
            for m in range(mat.size):
                mat.data[m] = b.flat[m]

    # Initialize activity
    for layer_name, activity in nvmnet.activity.items():
        output = syngen_net.get_neuron_data("nvm", layer_name, "output")
        if not isinstance(output, VoidArray):
            for m in range(output.size):
                output.data[m] = activity.flat[m]

    # Bias
    output = syngen_net.get_neuron_data("nvm", "bias", "output")
    if not isinstance(output, VoidArray):
        for m in range(output.size):
            output.data[m] = 1.0

    # Initialize comparison true pattern
    true_state = syngen_net.get_neuron_data("nvm", "co", "true_state")
    co = nvmnet.layers["co"]
    co_true = co.coder.encode("true")
    if not isinstance(true_state, VoidArray):
        for m in range(true_state.size):
            true_state.data[m] = co.activator.g(co_true.flat[m])


tick = 0
do_print = False

def make_syngen_environment(nvmnet, run_nvm=False,
        viz_layers=[], print_layers=[], stat_layers=[], read=True):
    global tick, do_print, foo
    tick = 0
    do_print = False

    ### INITIALIZE ENVIRONMENT ###
    layer_names = nvmnet.layers.keys() # deterministic order
    layer_names.remove("gh") # make sure hidden gates come first
    layer_names.insert(0,"gh")

    modules = [
        {
            "type" : "visualizer",
            "layers" : [
                {"structure": "nvm", "layer": layer_name}
                    for layer_name in viz_layers]
        },
    ]

    if read:
        modules.append({
            "type" : "callback",
            "layers" : [
                {
                    "structure" : "nvm",
                    "layer" : layer_name,
                    "output" : True,
                    "function" : "nvm_read",
                    "id" : i
                } for i,layer_name in enumerate(layer_names)
            ]
        })

    def read_callback(ID, size, ptr):
        global tick, do_print

        if run_nvm and ID == 0: #len(layer_names)-1:
            if do_print: print("nvm tick")
            nvmnet.tick()

        layer_name = layer_names[ID]

        ### Start printing current iteration
        if ID == 0:
            tick += 1
            
            coder = nvmnet.layers["gh"].coder
            h = np.array(FloatArray(size,ptr).to_list()).reshape(nvmnet.layers["gh"].shape)
            do_print = (coder.decode(h) == "ready")
            do_print = True

            if do_print and len(print_layers) > 0:
                print("Tick %d"%tick)
                if run_nvm:
                    print("Layer tokens (syngen|py)")
                else:
                    print("Layer tokens")
    
        if do_print and layer_name in print_layers:
            coder = nvmnet.layers[layer_name].coder
            syn_v = np.array(FloatArray(size,ptr).to_list())
            syn_tok = coder.decode(syn_v)
            if run_nvm:
                py_v = nvmnet.activity[layer_name]
                py_tok = coder.decode(py_v)
                residual = np.fabs(syn_v.reshape(py_v.shape) - py_v).max()
                print("%4s: %12s %s %12s (res=%f)"%(
                    layer_name,
                    syn_tok, "|" if syn_tok == py_tok else "X", py_tok, residual))
            else:
                print("%4s: %12s"%(layer_name, syn_tok))

            if layer_name == "opc" and syn_tok == "exit":
                print("DONE")
                interrupt_engine()
                    
        if do_print and layer_name in stat_layers:
            v = np.array(FloatArray(size,ptr).to_list())
            print("%s (syngen): %f, %f, %f"%(
                layer_name, v.min(), v.max(), np.fabs(v).mean()))
            if run_nvm:
                v = nvmnet.activity[layer_name]
                print("%s (py): %f, %f, %f"%(
                    layer_name, v.min(), v.max(), np.fabs(v).mean()))
                    
    create_io_callback("nvm_read", read_callback)

    return modules

'''
def init_syngen_nvm(nvmnet, syngen_net):
    connections = []

    for (to_name, from_name), w in nvmnet.weights.items():
        # Skip comparison circuitry
        if to_name != 'co' and from_name != 'ci':
            # weights
            mat_name = build_name(from_name, to_name, "-input")
            mat = syngen_net.get_weight_matrix(mat_name)
            if not isinstance(mat, VoidArray):
                for m in range(mat.size):
                    mat.data[m] = w.flat[m]
                connections.append((mat_name, np.size(w)))

            # biases
            b = nvmnet.biases[(to_name, from_name)]
            mat_name = build_name(from_name, to_name, "-biases")
            mat = syngen_net.get_weight_matrix(mat_name)
            if not isinstance(mat, VoidArray):
                for m in range(mat.size):
                    mat.data[m] = b.flat[m]
                connections.append((mat_name, np.size(w)))

    connections = sorted(connections, key = lambda x : x[1])
    for name,size in connections:
        print("%20s %10d" % (name, size))


if __name__ == "__main__":

    np.set_printoptions(linewidth=200, formatter = {'float': lambda x: '% .2f'%x})

    # nvmnet = make_saccade_nvm("tanh")
    nvmnet = make_saccade_nvm("logistic")

    print(nvmnet.layers["gh"].activator.off)
    print(nvmnet.w_gain, nvmnet.b_gain)
    print(nvmnet.layers["go"].activator.label)
    print(nvmnet.layers["gh"].activator.label)
    raw_input("continue?")

    structure, connections = make_syngen_network(nvmnet)
    modules = make_syngen_environment(nvmnet,
        initial_patterns = dict(nvmnet.activity),
        run_nvm=False,
        viz_layers = ["sc","fef","tc","ip","opc","op1","op2","gh","go"],
        print_layers = nvmnet.layers,
        # stat_layers=["ip","go","gh"])
        stat_layers=[],
        read=True)

    net = Network({"structures" : [structure], "connections" : connections})
    env = Environment({"modules" : modules})

    init_syngen_nvm(nvmnet, net)

    print(net.run(env, {"multithreaded" : True,
                            "worker threads" : 0,
                            "iterations" : 200,
                            "refresh rate" : 0,
                            "verbose" : True,
                            "learning flag" : False}))
    
    # Delete the objects
    del net

    # show_layers = [
    #     ["go", "gh","ip"] + ["op"+x for x in "c123"] + ["d0","d1","d2"],
    # ]
    # show_tokens = True
    # show_corrosion = True
    # show_gates = False

    # for t in range(100):
    #     # if True:
    #     # if t % 2 == 0 or at_exit:
    #     if nvmnet.at_start() or nvmnet.at_exit():
    #         print('t = %d'%t)
    #         print(nvmnet.state_string(show_layers, show_tokens, show_corrosion, show_gates))
    #     if nvmnet.at_exit():
    #         break
    #     nvmnet.tick()
'''
