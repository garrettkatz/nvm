from gate_sequencer import GateSequencer

def gcopy(to_layer, from_layer):
    """gates for inter-layer signal"""
    return [(to_layer,from_layer,'u'), (to_layer,to_layer,'d')]

def gprog():
    """gates for program memory (ip and op) layer updates"""
    return [('ip','ip','u')]+[k for x in ['c','1','2','3'] for k in gcopy("op"+x, 'ip')]

def flash_instruction_set(nvmnet):
    """
    layers: dict of layers, including:
        gate_output, gate_hidden, ip, opc, op1, op2, op3, cmph, cmpo
    """
    
    gate_map, layers, devices = nvmnet.gate_map, nvmnet.layers, nvmnet.devices
    gate_output, gate_hidden = layers["gate_output"], layers["gate_hidden"]
    gs = GateSequencer(gate_map, gate_output, gate_hidden, layers)

    ### Start executing new instruction
    
    # Load operands and step instruction pointer
    h = gate_hidden.coder.encode("start")
    g = gs.make_gate_output(ungate = gprog())
    gate_output.coder.encode("start", g)
    g_start, h_start = g.copy(), h.copy()    
    
    # Let opcode bias the gate layer
    g, h = gs.add_transit(ungate = [("gate_hidden","opc","u")],
        old_gates = g, old_hidden=h)

    # Ready to execute instruction
    h_ready = h.copy()
    g_ready = g.copy()
    gate_hidden.coder.encode('ready', h_ready)
    gate_output.coder.encode('ready', g_ready)
    
    ###### NOP
    
    # just return to start state
    g, h = gs.add_transit(
        ungate = gprog(),
        new_hidden = h_start,
        old_gates = g_ready, old_hidden = h_ready, opc = "nop")

    ###### SET

    # Let op1 bias the gate layer
    g_set, h_set = gs.add_transit(ungate = [("gate_hidden","op1","u")],
        old_gates = g_ready, old_hidden = h_ready, opc="set")
    gate_hidden.coder.encode('set', h_set)
    gate_output.coder.encode('set', g_set)    

    for name, device in devices.items():
        # Open flow from op2 to device in op1
        g, h = gs.add_transit(
            ungate = gcopy(name, "op2"),
            old_gates = g_set, old_hidden = h_set, op1 = name)
        gate_hidden.coder.encode('set_'+name, h)
        gate_output.coder.encode('set_'+name, g)    

        # return to start state
        g, h = gs.add_transit(
            ungate = gprog(),
            new_hidden = h_start,
            old_gates = g, old_hidden = h)

    weights, biases = gs.flash()
    return weights, biases
