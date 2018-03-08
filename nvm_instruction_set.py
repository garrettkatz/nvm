from gate_sequencer import GateSequencer

def gcopy(to_layer, from_layer):
    """gates for inter-layer signal"""
    return [(to_layer,from_layer,'u'), (to_layer,to_layer,'d')]

def gprog():
    """gates for program memory (ip and op) layer updates"""
    return [('ip','ip','u')]+[k for x in ['c','1','2','3'] for k in gcopy("op"+x, 'ip')]

def flash_instruction_set(nvm):
    """
    layers: dict of layers, including:
        gate_output, gate_hidden, ip, opc, op1, op2, op3, cmph, cmpo
    """
    
    gate_map, layers = nvm.gate_map, nvm.layers
    gate_output, gate_hidden = layers["gate_output"], layers["gate_hidden"]
    gs = GateSequencer(gate_map, gate_output, gate_hidden, layers)

    ### Start executing new instruction

    g_start = gs.make_gate_output()
    gate_output.coder.encode("off", g_start)
    h_start = gate_hidden.coder.encode("start")
    
    # load ops from program memory and step program memory
    g, h = gs.add_transit(ungate = gprog(), old_gates = g_start, old_hidden = h_start)
    gate_output.coder.encode("load op",g)
    gate_hidden.coder.encode("load op",h)
    
    # stabilize memory
    g, h = gs.stabilize(h)
    gate_hidden.coder.encode("stable",h)
    # gate_output.coder.encode("stable",g)
    
    g, h = gs.add_transit(
        new_hidden = h_start, # begin next clock cycle
        old_gates = g, old_hidden = h)
    
    # # Let opcode bias the gate layer
    # g, h = gs.add_transit(ungate = [("gate_hidden","opc","u")], old_hidden=h)
    # gate_output.coder.encode('op bias', g)
    
    # # Ready to execute instruction
    # h_ready = h.copy()
    # g_ready = g.copy()
    # gate_hidden.coder.encode('ready', h_ready)
    # gate_output.coder.encode('ready', g_ready)
    
    # ###### NOP instruction
    
    # g, h = gs.add_transit(
    #     new_hidden = h_start, # begin next clock cycle
    #     old_gates = g_ready, old_hidden = h_ready, opc = "nop")

    weights, biases = gs.flash()
    return weights, biases
