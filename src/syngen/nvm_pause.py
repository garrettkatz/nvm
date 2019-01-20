import numpy as np
from nvm_net import make_nvmnet

if __name__ == '__main__':

    np.set_printoptions(linewidth=200, formatter = {'float': lambda x: '% .2f'%x})
   
    nvmnet = make_nvmnet(programs = {"amy":"""
    
    loop:   mov d0 A
            mov d0 B
            jmp loop
    
    """})

    # gh <- gh pauses, but go <- gh remains so as to resume when amy dies down
    gate_output, gate_hidden = nvmnet.layers['go'], nvmnet.layers['gh']
    keep_on = [
        (gate_output.name, gate_hidden.name, 'u'),
        (gate_output.name, gate_output.name, 'd')]
    
    # Input that would turn on provided gates
    gate_map = nvmnet.gate_map
    w_amy = -np.ones((gate_output.size, 1))
    for k in keep_on:
        i = gate_map.get_gate_index(k)
        w_amy[i,0] = +1
        print("gate %s: %d of %d"%(k, i, gate_output.size))
    gate_output.coder.encode("pause", (w_amy > 0).astype(float))

    # Scale to outweigh any go <- gh signal
    scale = np.fabs(nvmnet.weights[("go","gh")]).sum(axis=1)[:,np.newaxis] + \
        np.fabs(nvmnet.biases[("go","gh")])
    w_amy *= scale
        
    show_layers = [
        ["go", "gh","ip"] + ["op"+x for x in "c12"] + ["d0","d1","d2"],
    ]
    show_tokens = True
    show_corrosion = False
    show_gates = False

    t_amy_on = 10
    t_amy_off = 15
    raw_input("Should pause from t=%d to %d. continue?"%(t_amy_on, t_amy_off))

    for t in range(20):

        if t_amy_on <= t and t < t_amy_off:
            nvmnet.activity["go"] = (w_amy > 0).astype(float)

        if True:
        # if t % 2 == 0 or nvmnet.at_exit():
        # if nvmnet.at_start() or nvmnet.at_exit():
            print('t = %d'%t)
            print(nvmnet.state_string(show_layers, show_tokens, show_corrosion, show_gates))
        if nvmnet.at_exit():
            break

        nvmnet.tick()
