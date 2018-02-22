import numpy as np
import sys
from tokens import N_LAYER, get_token, LAYERS, DEVICES, TOKENS
from flash_rom import V_START, V_READY
from gates import N_GH, get_open_gates, PAD, N_GATES
from aas_nvm import make_weights, store_program, nvm_synapto
from aas_nvm import tick as pytick
from syngen import Network, Environment, create_io_callback, FloatArray, get_cpu, set_debug

#set_debug(True)

# program
REG_INIT = {"TC": "NULL"}
program = [ # label opc op1 op2 op3
    # "NULL","NOP","NULL","NULL","NULL",
    # "NULL","NOP","NULL","NULL","NULL",
    "NULL","SET","REG2","FACE","NULL", # Store "face" flag for comparison with TC
    "NULL","SET","REG3","TRUE","NULL", # Store "true" for unconditional jumps
    # plan center saccade
    "REPEAT","SET","FEF","CENTER","NULL",
    # initiate saccade
    "NULL","SET","SC","SACCADE","NULL",
    "NULL","SET","SC","OFF","NULL",
    # wait face
    "LOOP","CMP","REG1","TC","REG2", # Check if TC detects face
    "NULL","JMP","REG1","LOOK","NULL", # If so, skip to saccade step
    "NULL","JMP","REG3","LOOP","NULL", # Check for face again
    # initiate saccade
    "LOOK","SET","SC","SACCADE","NULL", # TC detected gaze, allow saccade
    "NULL","SET","SC","OFF","NULL", # TC detected gaze, allow saccade
    # repeat
    "NULL","JMP","REG3","REPEAT","NULL", # Successful saccade, repeat
]


# create weights
weights = make_weights()
weights, v_prog = store_program(weights, program, do_global=True)
structures, connections = nvm_synapto(weights)

# initialize py activity
ACTIVITY = {k: -PAD*np.ones((N_LAYER,1)) for k in LAYERS+DEVICES}
ACTIVITY["GATES"] = V_START[:N_GH,:]
ACTIVITY["MEM"] = v_prog[:N_LAYER,[0]]
ACTIVITY["MEMH"] = v_prog[N_LAYER:,[0]]
ACTIVITY["CMPA"] = PAD*np.sign(np.random.randn(N_LAYER,1))
ACTIVITY["CMPB"] = -ACTIVITY["CMPA"]
for k,v in REG_INIT.items(): ACTIVITY[k] = TOKENS[v]

init_layers = ["MEM", "MEMH", "GATES"]
pad_init_layers = [l for l in LAYERS + DEVICES if l not in init_layers]
callback_layers = ["GATES", "MEM", "OPC", "OP1", "OP2", "OP3"]

init_data = [v_prog.flat[:N_LAYER], v_prog.flat[N_LAYER:], V_START.flat[:N_GH]]

def init_callback(ID, size, ptr):
    arr = FloatArray(size,ptr)
    if ID < len(init_layers):
        print("Initializing " + init_layers[ID])
        for i,x in enumerate(init_data[ID]):
            arr.data[i] = np.arctanh(x)
        if ID == 2:
            print("gate init:")
            gate_output = np.array(FloatArray(size,ptr).to_list())[:,np.newaxis]
            #do_print = (gate_output * V_READY[:N_GH,:] >= 0).all()
            do_print = True
            if do_print:
                print("Gates: ", get_open_gates(gate_output))
                print("min, max, avg abs init")
                print(gate_output[:N_GATES,:].min())
                print(gate_output[:N_GATES,:].max())
                print(np.fabs(gate_output[:N_GATES,:]).mean())
    else:
        print("Initializing " + pad_init_layers[ID-len(init_layers)])
        for i in xrange(size):
            arr.data[i] = -np.arctanh(PAD)

tick = 0
do_print = False
def read_callback(ID, size, ptr):
    global tick, do_print, ACTIVITY, weights

    if tick == 40: sys.exit()

    if ID == 0:
        tick += 1

        gate_output = np.array(FloatArray(size,ptr).to_list())[:,np.newaxis]
        #do_print = (gate_output * V_READY[:N_GH,:] >= 0).all()
        do_print = True
        if do_print:
            print("Tick (syngen): " + str(tick))
            print("Gates: ", get_open_gates(gate_output))
            print("min, max, avg abs syn")
            print(gate_output[:N_GATES,:].min())
            print(gate_output[:N_GATES,:].max())
            print(np.fabs(gate_output[:N_GATES,:]).mean())

        # side by side py
        gate_output = ACTIVITY["GATES"]
        if do_print:
            print("Tick (py): " + str(tick))
            print("Gates: ", get_open_gates(gate_output))
            # print("min, max, avg abs py")
            # print(gate_output[:N_GATES,:].min())
            # print(gate_output[:N_GATES,:].max())
            # print(np.fabs(gate_output[:N_GATES,:]).mean())
        ACTIVITY = pytick(ACTIVITY, weights)

    else:
        if do_print:
            print(callback_layers[ID], get_token(FloatArray(size,ptr).to_list()))
            if ID == len(callback_layers)-1:
                print("")
    
init_cb,init_addr = create_io_callback(init_callback)
read_cb,read_addr = create_io_callback(read_callback)


# Create network
network = Network(
    {"structures" : structures,
     "connections" : connections})

# Set non-trivial weights (row or column order?)
for h in ["","H"]:
    mat_name ="MEM"+h + "<MEMH-input"
    mat = network.get_weight_matrix(mat_name)
    print(mat.size, mat_name)
    for i in range(mat.size):
        mat.data[i] = weights[("MEM"+h, "MEMH")].flat[i]
for from_layer in ["GATES","OPC","OP1","OP2","OP3"]:
    mat_name = "GATES<"+from_layer+"-input"
    mat = network.get_weight_matrix(mat_name)
    print(mat.size, mat_name)
    for i in range(mat.size):
        mat.data[i] = weights[("GATES", from_layer)].flat[i]

# Create environment modules
modules = [
    {
        "type" : "visualizer",
        "layers" : [
            {"structure": "nvm", "layer": layer}
                for layer in ["MEM","MEMH","GATES"]]
    },
    {
        "type" : "periodic_input",
        "value" : 1.0,
        "layers" : [{"structure": "nvm", "layer": "bias"}]
    },
#    {
#        "type" : "csv_output",
#        "cutoff" : 1,
#        "layers" : [{"structure": "nvm", "layer": "MEM"}]
#    },
    {
        "type" : "callback",
        "cutoff" : "1",
        "layers" : [
            {
                "structure" : "nvm",
                "layer" : layer,
                "input" : "true",
                "function" : init_addr,
                "id" : i
            } for i,layer in enumerate(init_layers + pad_init_layers)
        ]
    },
    {
        "type" : "callback",
        "layers" : [
            {
                "structure" : "nvm",
                "layer" : layer,
                "output" : "true",
                "function" : read_addr,
                "id" : i
            } for i,layer in enumerate(callback_layers)
        ]
    }

#    {
#        "type" : "saccade",
#        "layers" : [{"structure": "nvm", "layer": layer} for layer in [
#            "MEM","MEMH","GATES"]]
#    },
]
env = Environment({"modules" : modules})
device = get_cpu()
print(network.run(env, {"multithreaded" : "true",
                        "worker threads" : 1,
                        "iterations" : 0,
                        "refresh rate" : 5,
                        "devices" : device,
                        "verbose" : "true",
                        "learning flag" : "false"}))


# Delete the objects
del network
