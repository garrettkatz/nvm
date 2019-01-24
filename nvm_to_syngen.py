import numpy as np
from saccade_programs import make_saccade_nvm
from syngen import Network, Environment, ConnectionFactory, create_io_callback, FloatArray, set_debug, VoidArray

# Builds a name for a connection or dendrite
def build_name(from_name, to_name, suffix=""):
    return to_name + "<" + from_name + suffix

def make_syngen_network(nvmnet):
    
    ### SET UP NETWORK CONFIG ###
    
    layer_configs = []

    for layer_name, layer in nvmnet.layers.items():
        # Add dendrites
        dendrites = []
        for (to_name, from_name) in nvmnet.weights:
            if to_name == layer_name:
                dendrites.append({"name" : build_name(from_name, to_name)})

        # Add dendrites for update/decay gain
        dendrites.append(
            {
                "name" : "gain",
                "children" : [
                    {
                        "name" : "gain-update",
                        "opcode" : "add",
                        "init" : 1.0
                    },
                    {
                        "name" : "gain-decay",
                        "opcode" : "mult",
                        "init" : 1.0
                    },
                    {
                        "name" : "fix",
                        "opcode" : "mult",
                        "init" : nvmnet.b_gain
                    },
                ]
            }
        )

        # Build layer config
        layer_configs.append({
            "name" : layer_name,
            "neural model" : "nvm_" + layer.activator.label,
            "dendrites" : dendrites,
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

    # Comparison SOM
    layer_configs.append({
        "name" : "csom",
        "neural model" : "som",
        "rows" : 1,
        "columns" : 1,
        "rbf scale" : 1,
    })

    # Parameters shared by all connections
    defaults = { "plastic" : False }

    # Build fully connected connection
    def build_full_biases(from_layer, to_layer, props):
        props["name"] = build_name(from_layer.name, to_layer.name, "-biases")
        props["from layer"] = "bias"
        props["to layer"] = to_layer.name
        props["dendrite"] = build_name(from_layer.name, to_layer.name)
        props["type"] = "fully connected"
        props["opcode"] = "add"
        props["sparse"] = False
        props["weight config"] = {
            "type" : "flat",
            "weight" : 0.0  # This will get overwritten when the ROM is flashed
        }
    def build_full_weights(from_layer, to_layer, props):
        props["name"] = build_name(from_layer.name, to_layer.name, "-input")
        props["from layer"] = from_layer.name
        props["to layer"] = to_layer.name
        props["dendrite"] = build_name(from_layer.name, to_layer.name)
        props["type"] = "fully connected"
        props["opcode"] = "add"
        props["sparse"] = False
        props["weight config"] = {
            "type" : "flat",
            "weight" : 0.0  # This will get overwritten when the ROM is flashed
        }

    # Build gate for another connection
    def build_update(from_layer, to_layer, props):
        props["name"] = build_name(from_layer.name, to_layer.name, "-input-update")
        props["from layer"] = "go" # gate output
        props["to layer"] = to_layer.name
        props["dendrite"] = build_name(from_layer.name, to_layer.name)
        props["opcode"] = "mult" #_heaviside"
        props["sparse"] = False
        props["weight config"] = {
            "type" : "flat",
            "weight" : 1
        }
        props["type"] = "subset"

        gate_index = nvmnet.gate_map.get_gate_index((to_layer.name, from_layer.name, "u"))
        props["subset config"] = {
            "from row start" : 0,
            "from row end" : 1,
            "from column start" : gate_index,
            "from column end" : gate_index+1,
            "to row start" : 0,
            "to row end" : to_layer.shape[0],
            "to column start" : 0,
            "to column end" : to_layer.shape[1]
        }

    # Builds gain self-connections (from_layer = to_layer)
    def build_gain(from_layer, to_layer, props):
        props["name"] = build_name(from_layer.name, to_layer.name, "-gain")
        props["from layer"] = from_layer.name
        props["to layer"] = to_layer.name
        props["dendrite"] = "fix"
        props["type"] = "one to one"
        props["opcode"] = "add"
        props["sparse"] = False
        props["weight config"] = {
            "type" : "flat",
            "weight" : nvmnet.w_gain
        }

    # Builds update gate of gain (from_layer = gate output)
    def build_gain_update(from_layer, to_layer, props):
        props["name"] = build_name(from_layer.name, to_layer.name, "-update")
        props["from layer"] = from_layer.name
        props["to layer"] = to_layer.name
        props["dendrite"] = "gain-update"
        props["opcode"] = "sub" # dendrite has bias, attributes have heaviside
        props["sparse"] = False
        props["weight config"] = {
            "type" : "flat",
            "weight" : 1
        }
        props["type"] = "subset"

        update_gate_index = nvmnet.gate_map.get_gate_index(
            (to_layer.name, to_layer.name, "u"))
        props["subset config"] = {
            "from row start" : 0,
            "from row end" : 1,
            "from column start" : update_gate_index,
            "from column end" : update_gate_index+1,
            "to row start" : 0,
            "to row end" : to_layer.shape[0],
            "to column start" : 0,
            "to column end" : to_layer.shape[1],
        }

    # Builds decay gate of gain (from_layer = gate output)
    def build_gain_decay(from_layer, to_layer, props):
        props["name"] = build_name(from_layer.name, to_layer.name,"-decay")
        props["from layer"] = from_layer.name
        props["to layer"] = to_layer.name
        props["dendrite"] = "gain-decay"
        props["opcode"] = "sub" # dendrite has bias, attributes have heaviside
        props["sparse"] = False
        props["weight config"] = {
            "type" : "flat",
            "weight" : 1
        }
        props["type"] = "subset"

        decay_gate_index = nvmnet.gate_map.get_gate_index(
            (to_layer.name, to_layer.name, "d"))
        props["subset config"] = {
            "from row start" : 0,
            "from row end" : 1,
            "from column start" : decay_gate_index,
            "from column end" : decay_gate_index+1,
            "to row start" : 0,
            "to row end" : to_layer.shape[0],
            "to column start" : 0,
            "to column end" : to_layer.shape[1],
        }

    # Create factories
    full_biases_factory = ConnectionFactory(defaults, build_full_biases)
    full_weights_factory = ConnectionFactory(defaults, build_full_weights)
    update_factory = ConnectionFactory(defaults, build_update)
    gain_factory = ConnectionFactory(defaults, build_gain)
    gain_update_factory = ConnectionFactory(defaults, build_gain_update)
    gain_decay_factory = ConnectionFactory(defaults, build_gain_decay)

    connections = []

    # Build standard connections and their gates
    for (to_name, from_name) in nvmnet.weights:
        to_layer, from_layer = nvmnet.layers[to_name], nvmnet.layers[from_name]

        # Special comparison circuitry
        if to_name != 'co' or from_name != 'ci':
            # u * (w v + b)
            connections.append(full_biases_factory.build(from_layer, to_layer))
            connections.append(full_weights_factory.build(from_layer, to_layer))
            connections.append(update_factory.build(from_layer, to_layer))
        else:
            coder = nvmnet.layers['co'].coder
            act = nvmnet.layers['co'].activator

            # ci to csom
            connections.append({
                "name" : "csom<ci",
                "from layer" : from_layer.name,
                "to layer" : "csom",
                "type" : "fully connected",
                "opcode" : "add",
                "sparse" : False,
                "plastic" : True,
                "learning rate" : 1,
                "weight config" : {
                    "type" : "flat",
                    "weight" : 0.0  # This will get overwritten when CMP runs
                }
            })

            # csom to co
            co_weights = [act.g(x) - act.g(y) for x,y in
                zip(coder.encode('true').flatten(), coder.encode('false').flatten())]
            connections.append({
                "name" : "co<csom",
                "from layer" : "csom",
                "to layer" : to_layer.name,
                "type" : "fully connected",
                "opcode" : "add",
                "sparse" : False,
                "plastic" : False,
                "weight config" : {
                    "type" : "specified",
                    "weight string" : " ".join(str(x) for x in co_weights)
                }
            })

            # co update gate
            #co_update = update_factory.build(from_layer, to_layer)
            #del co_update["dendrite"]
            #connections.append(co_update)

            # bias to co
            co_bias = full_biases_factory.build(from_layer, to_layer)
            co_bias["name"] = "co-bias"
            del co_bias["dendrite"]
            co_bias["weight config"] = {
                "type" : "specified",
                "weight string" : " ".join(str(act.g(x)) for x in
                    coder.encode('false').flatten())
            }
            connections.append(co_bias)

            # learning gate for csom
            gate_index = nvmnet.gate_map.get_gate_index(("co", "ci", "l"))
            connections.append({
                "name" : "csom-learning",
                "from layer" : "go", # gate output
                "to layer" : "csom",
                "opcode" : "modulate",
                "sparse" : False,
                "plastic" : False,
                "weight config" : {
                    "type" : "flat",
                    "weight" : 1
                },
                "type" : "subset",
                "subset config" : {
                    "from row start" : 0,
                    "from row end" : 1,
                    "from column start" : gate_index,
                    "from column end" : gate_index+1,
                    "to row start" : 0,
                    "to row end" : 1,
                    "to column start" : 0,
                    "to column end" : 1
                }
            })

    # Add gain connections
    gate_output = nvmnet.layers["go"]
    for layer in nvmnet.layers.values():
        # (1-u) * (1-d) * (w_gain v + b_gain)
        connections.append(gain_update_factory.build(gate_output, layer))
        connections.append(gain_decay_factory.build(gate_output, layer))
        connections.append(gain_factory.build(layer, layer))

    # Set structures
    for conn in connections:
        conn["from structure"] = "nvm"
        conn["to structure"] = "nvm"


    # Remove unused memory connections
    exclude = ["mf", "mb", "sf", "sb"]
    connections = [conn for conn in connections
        if conn["to layer"] not in exclude and conn["from layer"] not in exclude]

    exclude = ["op2", "fef"]
    connections = [conn for conn in connections
        if conn["to layer"] != "ip" or conn["from layer"] not in exclude]

    # Remove unused device connections
    devices = ["tc", "fef", "sc"]
    connections = [conn for conn in connections
        if conn["to layer"] not in devices or \
           conn["from layer"] not in devices or \
           conn["from layer"] == conn["to layer"]]

    # Remove unused op2 -> device connections
    include = ["fef"]
    connections = [conn for conn in connections
        if conn["from layer"] != "op2" or \
           conn["to layer"] not in devices or \
           conn["to layer"] in include]

    # Removed unused device -> ip connections
    include = ["tc"]
    connections = [conn for conn in connections
        if conn["to layer"] != "ip" or \
           conn["from layer"] not in devices or \
           conn["from layer"] in include]

    # Removed unused device -> ci connections
    include = ["tc"]
    connections = [conn for conn in connections
        if conn["to layer"] != "ci" or \
           conn["from layer"] not in devices or \
           conn["from layer"] in include]

    # Remove corresponding bias connections
    biases = [conn["name"].replace("input", "biases")
        for conn in connections if conn["name"].endswith("input")]
    connections = [conn for conn in connections
        if not conn["name"].endswith("biases") or conn["name"] in biases]

    # Remove unnecessary dendrites
    dendrites = [(conn["to layer"], conn["dendrite"])
        for conn in connections
            if "dendrite" in conn and conn["from layer"] != "go"]

    for layer in layer_configs:
        if "dendrites" in layer:
            layer["dendrites"] = [d
                for d in layer["dendrites"]
                    if d["name"] == "gain" or (layer["name"], d["name"]) in dendrites]

    # Remove unnecessary gates
    connections = [conn for conn in connections
        if "dendrite" not in conn or \
            "gain" in conn["dendrite"] or \
            (conn["to layer"], conn["dendrite"]) in dendrites]

    # Remove dummy 'di' connections (only necessary to instantiate gate)
    connections = [conn for conn in connections
        if conn["to layer"] != "di" and conn["from layer"] != "di"]

    structure = {"name" : "nvm",
                 "type" : "parallel",
                 "layers": layer_configs}

    return structure, connections

tick = 0
do_print = False

def make_syngen_environment(nvmnet, initial_patterns={}, run_nvm=False,
        viz_layers=[], print_layers=[], stat_layers=[], read=True):
    global tick, to_print
    tick = 0
    do_print = False

    ### INITIALIZE ENVIRONMENT ###
    layer_names = nvmnet.layers.keys() # deterministic order
    layer_names.remove("gh") # make sure hidden gates come first
    layer_names.insert(0,"gh")

    # Randomize TC input
    # If True, run the input callback indefinitely
    # Otherwise, only run it once
    random_tc = False

    modules = [
        {
            "type" : "visualizer",
            "layers" : [
                {"structure": "nvm", "layer": layer_name}
                    for layer_name in viz_layers]
        },
        {
            "type" : "callback",
            "cutoff" : 0 if random_tc else 1,
            "layers" : [
                {
                    "structure" : "nvm",
                    "layer" : layer_name,
                    "input" : True,
                    "function" : "nvm_input",
                    "id" : i,
                } for i,layer_name in enumerate(layer_names)
            ]
        }
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

    def input_callback(ID, size, ptr):

        layer_name = layer_names[ID]
    
        ### initialization
        if tick == 0:
        
            arr = FloatArray(size,ptr)
            act = nvmnet.layers[layer_name].activator

            if layer_name in initial_patterns:
                for i,x in enumerate(initial_patterns[layer_name]):
                    # Hack: multiply initial inputs by 10, pushing them further
                    #   from the origin and washing out bias corruption
                    # TODO: fix this in a less hacky way
                    # arr.data[i] = act.g(x) * 10
                    arr.data[i] = act.g(x) - nvmnet.b_gain # slightly less hacky

            else:
                for i in xrange(size):
                    arr.data[i] = act.g(act.off) - nvmnet.b_gain

        else:
            arr = FloatArray(size,ptr)

            # occassionally change tc
            tc_sched = [60, 110, 120]
            if layer_name == "tc" and tick % tc_sched[2] in tc_sched[:2]:
    
                if tick % tc_sched[2] == tc_sched[0]:
                    tok = ["left","right"][np.random.randint(2)] # face appears
                if tick % tc_sched[2] == tc_sched[1]:
                    tok = "wait" # face disappears
                    
                pattern = nvmnet.layers[layer_name].coder.encode(tok)
    
                act = nvmnet.layers[layer_name].activator
                for i,x in enumerate(pattern):
                    arr.data[i] = act.g(x)*10 # *10 to make sure previous state is wiped
    
                if run_nvm:
                    nvmnet.activity[layer_name] = pattern
                    
            else:
            
                for i in xrange(size):
                    arr.data[i] = 0.

    
    def read_callback(ID, size, ptr):
        global tick, do_print

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
                    
        if do_print and layer_name in stat_layers:
            v = np.array(FloatArray(size,ptr).to_list())
            print("%s (syngen): %f, %f, %f"%(
                layer_name, v.min(), v.max(), np.fabs(v).mean()))
            if run_nvm:
                v = nvmnet.activity[layer_name]
                print("%s (py): %f, %f, %f"%(
                    layer_name, v.min(), v.max(), np.fabs(v).mean()))
                    
        if run_nvm and ID == len(layer_names)-1:
            if do_print: print("nvm tick")
            nvmnet.tick()

    create_io_callback("nvm_input", input_callback)
    create_io_callback("nvm_read", read_callback)

    return modules

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
