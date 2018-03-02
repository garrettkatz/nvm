from gate_sequencer import GateSequencer, gmem, gcopy

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
    gate_output.coder.encode("start", g_start)
    h_start = gate_hidden.coder.encode("start")

    h = h_start
    _, h = gs.add_transit(new_hidden = h, old_hidden = h)
    
    # for reg in ["opc"]:#,"op1","op2","op3"]:
    #     # load op from memory and step memory
    #     _, h = gs.add_transit(ungate = gmem() + gcopy(reg,"mem"), old_hidden = h)
    #     # h = gs.stabilize(h)

    # # Let opcode bias the gate layer
    # g, h = gs.add_transit(ungate = [("ghide","opc","U")], old_hidden=h)
    
    # # Ready to execute instruction
    # h_ready = h.copy()
    
    # # _, _ = gs.add_transit(
    # #     new_hidden = h_start, # begin next clock cycle
    # #     old_gates = g, old_hidden = h_ready)

    
    # ###### NOP instruction
    
    # g, h = gs.add_transit(
    #     new_hidden = h_start, # begin next clock cycle
    #     old_gates = g, old_hidden = h_ready, opc = "nop")

    weights, bias = gs.flash()
    g_start = gs.make_gate_output()
    return weights, bias
