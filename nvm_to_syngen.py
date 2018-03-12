import numpy as np
from saccade_programs import make_saccade_nvm
from syngen import Network, Environment, ConnectionFactory, create_io_callback, FloatArray, set_debug

def nvm_to_syngen(nvmnet, initial_patterns={}, run_nvm=False, viz_layers=[], print_layers=[]):
    
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
                        "init" : 1.0 # bias
                    },
                    {
                        "name" : "gain-decay",
                        "opcode" : "mult",
                        "init" : 1.0 # bias
                    }
                ]
            }
        )

        # Build layer config
        layer_configs.append({
            "name" : layer_name,
            "neural model" : "nvm",
            "dendrites" : dendrites,
            "rows" : layer.shape[0],
            "columns" : layer.shape[1],
        })

    structures = [{"name" : "nvm",
                   "type" : "parallel",
                   "layers": layer_configs}]

    # Parameters shared by all connections
    defaults = { "plastic" : "false" }

    # Build fully connected connection
    def build_full(from_layer, to_layer, props):
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
        props["opcode"] = "mult_heaviside"
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

    # Builds a multiplicative gain connection (from_layer = to_layer)
    def build_gain(from_layer, to_layer, props):
        props["name"] = build_name(from_layer.name, to_layer.name, "-gain")
        props["from layer"] = from_layer.name
        props["to layer"] = to_layer.name
        props["dendrite"] = "gain"
        props["type"] = "one to one"
        props["opcode"] = "mult"
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
        props["opcode"] = "sub_heaviside"  # (1 - hs(u)), dendrite has bias
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
        props["opcode"] = "sub_heaviside"  # (1 - hs(d)), dendrite has bias
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
    full_factory = ConnectionFactory(defaults, build_full)
    update_factory = ConnectionFactory(defaults, build_update)
    gain_factory = ConnectionFactory(defaults, build_gain)
    gain_update_factory = ConnectionFactory(defaults, build_gain_update)
    gain_decay_factory = ConnectionFactory(defaults, build_gain_decay)

    connections = []

    # Build standard connections and their gates
    for (to_name, from_name) in nvmnet.weights:
        to_layer, from_layer = nvmnet.layers[to_name], nvmnet.layers[from_name]
        # u * (w v + b)
        connections.append(full_factory.build(from_layer, to_layer))
        connections.append(update_factory.build(from_layer, to_layer))

    # Add gain connections
    gate_output = nvmnet.layers["go"]
    for layer in nvmnet.layers.values():
        # (1-u) * (1-d) * (w_gain v + b_gain)
        connections.append(gain_factory.build(layer, layer))
        connections.append(gain_update_factory.build(gate_output, layer))
        connections.append(gain_decay_factory.build(gate_output, layer))

    ### INITIALIZE NETWORK AND WEIGHTS ###

    net = Network(
        {"structures" : structures,
         "connections" : connections})

    for (to_name, from_name), w in nvmnet.weights.items():
        mat_name = build_name(from_name, to_name, "-input")
        mat = net.get_weight_matrix(mat_name)
        for m in range(mat.size):
            mat.data[m] = w.flat[m]

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
            "cutoff" : "1",
            "layers" : [
                {
                    "structure" : "nvm",
                    "layer" : layer_name,
                    "input" : "true",
                    "function" : "nvm_init",
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

    def init_callback(ID, size, ptr):
        arr = FloatArray(size,ptr)
        layer_name = layer_names[ID]
        act = nvmnet.layers[layer_name].activator
        # print("Initializing " + layer_name)
        if layer_name in initial_patterns:
            for i,x in enumerate(initial_patterns[layer_name]):
                arr.data[i] = act.g(x)
        else:
            for i in xrange(size):
                arr.data[i] = act.g(act.off)

    def read_callback(ID, size, ptr):
        global tick, do_print

        layer_name = layer_names[ID]
    
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
                print("%4s: %7s %s %7s (res=%f)"%(
                    layer_name,
                    syn_tok, "|" if syn_tok == py_tok else "X", py_tok, residual))
            else:
                print("%4s: %7s"%(layer_name, syn_tok))
                    
        # if do_print and layer_name in print_layers:
        #     v = np.array(FloatArray(size,ptr).to_list())
        #     print("%s (syngen): %f, %f, %f"%(
        #         layer_name, v.min(), v.max(), np.fabs(v).mean()))
        #     if run_nvm:
        #         v = nvmnet.activity[layer_name]
        #         print("%s (py): %f, %f, %f"%(
        #             layer_name, v.min(), v.max(), np.fabs(v).mean()))
                    
        if run_nvm and ID == len(layer_names)-1:
            if do_print: print("nvm tick")
            nvmnet.tick()

    create_io_callback("nvm_init", init_callback)
    create_io_callback("nvm_read", read_callback)

    return net, env

if __name__ == "__main__":

    np.set_printoptions(linewidth=200, formatter = {'float': lambda x: '% .2f'%x})

    tick = 0
    do_print = False
   
    nvmnet = make_saccade_nvm()
    print(nvmnet.layers["gh"].activator.off)
    # raw_input("continue?")

    net, env = nvm_to_syngen(nvmnet,
        initial_patterns = dict(nvmnet.activity),
        run_nvm=True,
        viz_layers = ["ip","opc","op1","op2","go"],
        print_layers = nvmnet.layers,)

    print(net.run(env, {"multithreaded" : "true",
                            "worker threads" : 0,
                            "iterations" : 2,
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
