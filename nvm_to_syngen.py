import numpy as np
from saccade_programs import make_saccade_nvm
from syngen import Network, Environment, ConnectionFactory, create_io_callback, FloatArray, set_debug

def nvm_to_syngen(nvmnet, initial_patterns={}, run_nvm=False, viz_layers=[], print_layers=[], stat_layers=[]):
    
    # Builds a name for a connection or dendrite
    def build_name(from_name, to_name, suffix=""):
        return to_name + "<" + from_name + suffix
    
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

    structures = [{"name" : "nvm",
                   "type" : "parallel",
                   "layers": layer_configs}]

    # Parameters shared by all connections
    defaults = { "plastic" : "false" }

    # Build fully connected connection
    def build_full_biases(from_layer, to_layer, props):
        props["name"] = build_name(from_layer.name, to_layer.name, "-biases")
        props["from layer"] = "bias"
        props["to layer"] = to_layer.name
        props["dendrite"] = build_name(from_layer.name, to_layer.name)
        props["type"] = "fully connected"
        props["opcode"] = "add"
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
        # u * (w v + b)
        connections.append(full_biases_factory.build(from_layer, to_layer))
        connections.append(full_weights_factory.build(from_layer, to_layer))
        connections.append(update_factory.build(from_layer, to_layer))

    # Add gain connections
    gate_output = nvmnet.layers["go"]
    for layer in nvmnet.layers.values():
        # (1-u) * (1-d) * (w_gain v + b_gain)
        connections.append(gain_update_factory.build(gate_output, layer))
        connections.append(gain_decay_factory.build(gate_output, layer))
        connections.append(gain_factory.build(layer, layer))

    ### INITIALIZE NETWORK AND WEIGHTS ###

    net = Network(
        {"structures" : structures,
         "connections" : connections})

    for (to_name, from_name), w in nvmnet.weights.items():
        # weights
        mat_name = build_name(from_name, to_name, "-input")
        mat = net.get_weight_matrix(mat_name)
        for m in range(mat.size):
            mat.data[m] = w.flat[m]
        # biases
        b = nvmnet.biases[(to_name, from_name)]
        mat_name = build_name(from_name, to_name, "-biases")
        mat = net.get_weight_matrix(mat_name)
        for m in range(mat.size):
            mat.data[m] = b.flat[m]

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
        {
            "type" : "callback",
            "layers" : [
                {
                    "structure" : "nvm",
                    "layer" : layer_name,
                    "input" : "true",
                    "function" : "nvm_input",
                    "id" : i
                } for i,layer_name in enumerate(layer_names)
            ]
        },
        {
            "type" : "callback",
            "layers" : [
                {
                    "structure" : "nvm",
                    "layer" : layer_name,
                    "output" : "true",
                    "function" : "nvm_read",
                    "id" : i
                } for i,layer_name in enumerate(layer_names)
            ]
        }
    ]
    env = Environment({"modules" : modules})

    def input_callback(ID, size, ptr):

        layer_name = layer_names[ID]
    
        ### initialization
        if tick == 0:
        
            arr = FloatArray(size,ptr)
            act = nvmnet.layers[layer_name].activator

            if layer_name in initial_patterns:
                for i,x in enumerate(initial_patterns[layer_name]):
                    arr.data[i] = act.g(x)

            else:
                for i in xrange(size):
                    arr.data[i] = act.g(act.off)

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

            if do_print:
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

    return net, env

if __name__ == "__main__":

    np.set_printoptions(linewidth=200, formatter = {'float': lambda x: '% .2f'%x})

    tick = 0
    do_print = False
   
    nvmnet = make_saccade_nvm()
    print(nvmnet.layers["gh"].activator.off)
    raw_input("continue?")

    net, env = nvm_to_syngen(nvmnet,
        initial_patterns = dict(nvmnet.activity),
        run_nvm=True,
        viz_layers = ["sc","fef","tc","ip","opc","op1","op2","gh","go"],
        print_layers = nvmnet.layers,
        # stat_layers=["ip","go","gh"])
        stat_layers=[])

    print(net.run(env, {"multithreaded" : "true",
                            "worker threads" : 0,
                            "iterations" : 100,
                            "refresh rate" : 0,
                            "verbose" : "true",
                            "learning flag" : "false"}))
    
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
