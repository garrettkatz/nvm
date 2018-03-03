from gate_sequencer import GateSequencer, gprog, gcopy

def flash_instruction_set(nvm):
    """
    layers: dict of layers, including:
        gate_output, gate_hidden, mem, memh, opc, op1, op2, op3, cmph, cmpo
    """
    
    gate_map, layers = nvm.gate_map, nvm.layers
    gate_output, gate_hidden = layers["gate_output"], layers["gate_hidden"]
    gs = GateSequencer(gate_map, gate_output, gate_hidden, layers)

    ### Start executing new instruction

    g_start = gs.make_gate_output()
    gate_output.coder.encode("off", g_start)
    h_start = gate_hidden.coder.encode("start")
    
    # load ops from program memory and step program memory
    g, h = gs.add_transit(ungate = gprog(), old_hidden = h_start)
    gate_output.coder.encode("load op",g)
    gate_hidden.coder.encode("load op",h)
    
    # stabilize memory
    h = gs.stabilize(h)
    gate_hidden.coder.encode("stable",h)
    
    # Let opcode bias the gate layer
    g, h = gs.add_transit(ungate = [
        ("gate_hidden","op"+x,"u") for x in 'c123'], old_hidden=h)
    gate_output.coder.encode('op bias', g)
    
    # Ready to execute instruction
    h_ready = h.copy()
    gate_hidden.coder.encode('ready', h_ready)
    
    # # ###### NOP instruction
    
    # g, h = gs.add_transit(
    #     new_hidden = h_start, # begin next clock cycle
    #     old_gates = g, old_hidden = h_ready, opc = "nop")

    weights, bias = gs.flash()
    g_start = gs.make_gate_output()
    return weights, bias
