from gate_sequencer import GateSequencer

def gflow(to_layer, from_layer):
    """gates for inter-layer information flow"""
    return [(to_layer,from_layer,'u'), (to_layer,to_layer,'d')]

def gprog():
    """gates for program memory (ip and op) layer updates"""
    return [('ip','ip','u')]+[k for x in ['c','1','2'] for k in gflow('op'+x, 'ip')]

def flash_instruction_set(nvmnet):
    """
    layers: dict of layers, including:
        gate_output, gate_hidden, ip, opc, op1, op2, op3, cmph, cmpo
    """
    
    gate_map, layers, devices = nvmnet.gate_map, nvmnet.layers, nvmnet.devices
    gate_output, gate_hidden = layers['go'], layers['gh']
    gs = GateSequencer(gate_map, gate_output, gate_hidden, layers)

    ### Start executing new instruction
    
    # Load operands and step instruction pointer
    h = gate_hidden.coder.encode('start')
    g = gs.make_gate_output(ungate = gprog())
    gate_output.coder.encode('start', g)
    g_start, h_start = g.copy(), h.copy()    
    
    # Let opcode bias the gate layer
    g, h = gs.add_transit(ungate = [('gh','opc','u')],
        old_gates = g, old_hidden=h)

    # Ready to execute instruction
    g_ready, h_ready = g.copy(), h.copy()
    gate_hidden.coder.encode('ready', h_ready)
    gate_output.coder.encode('ready', g_ready)
    
    ###### NOP
    
    # just return to start state
    gs.add_transit(
        new_gates = g_start, new_hidden = h_start,
        old_gates = g_ready, old_hidden = h_ready, opc = 'nop')

    ###### MOVV

    # Let op1 bias the gate layer
    g, h = gs.add_transit(ungate = [('gh','op1','u')],
        old_gates = g_ready, old_hidden = h_ready, opc='movv')
    g_movv, h_movv = g.copy(), h.copy()
    gate_hidden.coder.encode('movv', h_movv)
    gate_output.coder.encode('op1', g_movv)

    for name, device in devices.items():
        # Open flow from op2 to device in op1
        g, h = gs.add_transit(
            ungate = gflow(name, 'op2'),
            old_gates = g_movv, old_hidden = h_movv, op1 = name)
        gate_hidden.coder.encode('movv_'+name, h)
        gate_output.coder.encode('movv_'+name, g)

        # return to start state
        gs.add_transit(
            new_gates = g_start, new_hidden = h_start,
            old_gates = g, old_hidden = h)

    ###### MOVD

    # Let op1 and op2 bias the gate layer
    g, h = gs.add_transit(ungate = [
        ('gh','op1','u'),('gh','op2','u')
        ],
        old_gates = g_ready, old_hidden = h_ready, opc='movd')
    g_movd, h_movd = g.copy(), h.copy()
    gate_hidden.coder.encode('movd', h_movd)
    gate_output.coder.encode('movd', g_movd)    

    for from_name, from_device in devices.items():
        for to_name, to_device in devices.items():
            # Open flow between devices
            g, h = gs.add_transit(
                ungate = gflow(to_name, from_name),
                old_gates = g_movd, old_hidden = h_movd,
                op1 = to_name, op2 = from_name)
            gate_hidden.coder.encode('movd_'+to_name+'_'+from_name, h)
            gate_output.coder.encode('movd_'+to_name+'_'+from_name, g)

            # return to start state
            gs.add_transit(
                new_gates = g_start, new_hidden = h_start,
                old_gates = g, old_hidden = h)

    ###### JIF

    # Let op1 bias the gate layer
    g, h = gs.add_transit(ungate = [('gh','op1','u')],
        old_gates = g_ready, old_hidden = h_ready, opc='jif')
    g_jif, h_jif = g.copy(), h.copy()
    gate_hidden.coder.encode('jif', h_jif)

    for device in devices:

        # Let device named by op1 bias the gate layer
        g, h = gs.add_transit(ungate = [('gh',device,'u')],
            old_gates = g_jif, old_hidden = h_jif, op1=device)
        g_jd, h_jd = g.copy(), h.copy()
        gate_hidden.coder.encode('jif_'+device, h_jd)
        gate_output.coder.encode('jif_'+device, g_jd)

        # If device contains false, just return to start state
        gs.add_transit(
            new_gates = g_start, new_hidden = h_start,
            old_gates = g_jd, old_hidden = h_jd,
            **{device: 'false'}) # kwarg is device, not 'device'

        # If device contains true, open flow from op2 to ip
        g, h = gs.add_transit(
            ungate = gflow('ip', 'op2'),
            old_gates = g_jd, old_hidden = h_jd,
            **{device: 'true'})
        g_jt, h_jt = g.copy(), h.copy()
        gate_hidden.coder.encode('jif_true', h_jt)
        gate_output.coder.encode('jif_true', g_jt)

        # Stabilize ip
        g, h = gs.stabilize(h, num_iters=5)

        # then return to start state
        gs.add_transit(
            new_gates = g_start, new_hidden = h_start,
            old_gates = g, old_hidden = h)

    ###### JMPV

    # Open flow from op1 to ip
    g, h = gs.add_transit(ungate = gflow('ip','op1'),
        old_gates = g_ready, old_hidden = h_ready, opc='jmpv')
    g_jmpv, h_jmpv = g.copy(), h.copy()
    gate_hidden.coder.encode('jmpv', h_jmpv)
    gate_output.coder.encode('ip<op1', g_jmpv)

    # Stabilize ip
    g, h = gs.stabilize(h, num_iters=3)

    # then return to start state
    gs.add_transit(
        new_gates = g_start, new_hidden = h_start,
        old_gates = g, old_hidden = h)

    ###### JMPD

    # Let op1 bias the gate layer
    g, h = gs.add_transit(ungate = [('gh','op1','u')],
        old_gates = g_ready, old_hidden = h_ready, opc='jmpd')
    g_jmpd, h_jmpd = g.copy(), h.copy()
    gate_hidden.coder.encode('jmpd', h_jmpd)

    for device in devices:

        # Open flow from device in op1 to ip
        g, h = gs.add_transit(
            ungate = gflow('ip', device),
            old_gates = g_jmpd, old_hidden = h_jmpd,
            op1 = device)
        g_jmpd_dev, h_jmpd_dev = g.copy(), h.copy()
        gate_hidden.coder.encode('jmpd_'+device, h_jmpd_dev)
        gate_output.coder.encode('jmpd_'+device, g_jmpd_dev)

        # Stabilize ip
        g, h = gs.stabilize(h, num_iters=3)

        # then return to start state
        gs.add_transit(
            new_gates = g_start, new_hidden = h_start,
            old_gates = g, old_hidden = h)

    weights, biases, residual = gs.flash()
    return weights, biases
