import sys, time
import multiprocessing as mp
import numpy as np
import scipy as sp
import visualizer as vz
import mock_net as mn

class NVM:
    def __init__(self, coding, network):
        self.coding = coding
        self.network = network
        self.visualizing = False
        # Encode layer names and constants
        for symbol in self.network.get_layer_names():
            self.coding.encode(symbol)
    def __str__(self):
        pattern_list = self.network.list_patterns()
        vmstr = ''
        for (layer_name, pattern) in pattern_list:
            if self.decode(pattern)=='<?>': continue
            vmstr += '%s:%s;'%(layer_name, self.decode(pattern))
        return vmstr
    def encode(self,human_readable):
        return self.coding.encode(human_readable)
    def decode(self, machine_readable):
        return self.coding.decode(machine_readable)
    def tick(self):
        # network update
        self.network.tick()
        # answer any visualizer request
        if self.visualizing:
            if self.viz_pipe.poll():
                # flush request
                self.viz_pipe.recv()
                # respond with data
                self.send_viz_data()
    def send_viz_data(self, down_sample=2):
        """
        Protocol:
            <# layers>, <name>, <value>, <pattern>, <name>, <value>, <pattern>, ...
        """
        if not self.visualizing: return
        pattern_list = self.network.list_patterns()
        self.viz_pipe.send(len(pattern_list))
        for (layer_name, pattern) in pattern_list:
            self.viz_pipe.send(layer_name)
            self.viz_pipe.send(self.decode(pattern)) # value
            # down sample pattern
            pattern = np.concatenate((pattern, np.nan*np.ones(len(pattern) % down_sample)))
            pattern = pattern.reshape((len(pattern)/down_sample, down_sample)).mean(axis=1)
            pattern = (128*(pattern + 1.0)).astype(np.uint8).tobytes()
            self.viz_pipe.send_bytes(pattern) # bytes
    def show(self):
        self.hide() # flush any windowless viz process
        self.viz_pipe, other_end = mp.Pipe()
        self.viz_process = mp.Process(target=run_viz, args=(other_end,))
        self.viz_process.start()
        self.visualizing = True
        # send initial data for window layout
        self.send_viz_data()
    def hide(self):
        if not self.visualizing: return
        self.viz_pipe.send('shutdown')
        self.viz_process.join()
        self.viz_pipe = None
        self.viz_process = None
        self.visualizing = False
    def set_standard_input(self, message, from_human_readable=True):
        if from_human_readable:
            pattern = self.encode(message)
        else:
            pattern = np.fromstring(pattern,dtype=float)
        self.network.set_pattern('STDI', pattern)
    def get_standard_output(self, to_human_readable=True):
        pattern = self.network.get_pattern('STDO')
        if to_human_readable:
            message = self.decode(pattern)
        else:
            message = pattern.tobytes()
        return message
    def set_instruction(self, opcode, *operands):
        # clear gates
        self.network.set_pattern('A',self.network.get_pattern('A')*0)
        # set instruction
        self.network.set_pattern('OPC',self.encode(opcode))
        for op in range(len(operands)):
            self.network.set_pattern('OP%d'%(op+1), self.encode(operands[op]))
    def train(self, pattern_hash, new_pattern_hash):
        # train module with module.train
        # self.network.get_module(module_name).train(pattern_list, next_pattern_list)
        self.network.train(pattern_hash, new_pattern_hash)
    def quit(self):
        self.hide()
        sys.exit(0)

def mock_nvm(num_registers=3, layer_size=32):
    layer_names = ['IP','OPC','OP1','OP2','OP3'] # instruction
    layer_names += ['{%d}'%r for r in range(num_registers)] # registers
    layer_names += ['C1','C2','CO','N1','N2','NO'] # compare+nand
    layer_names += ['K','V'] # memory
    layer_names += ['STDI','STDO'] # io
    layer_sizes = [layer_size]*len(layer_names)
    net = mn.MockNet(layer_names, layer_sizes)
    coding = mn.MockCoding(layer_size)
    return NVM(coding, net)

def run_viz(nvm_pipe):
    viz = vz.Visualizer(nvm_pipe)
    viz.launch()

def flash_nrom(vm):
    # train vm on instruction set
    # gate_index_map = vm.network.get_module('gating').gate_index_map
    omega = np.tanh(1)
    gate_pattern = vm.network.get_pattern('A')
    # get non-gate layer names
    layer_names = vm.network.get_layer_names(omit_gates=True)
    # layer copies
    zero_gate_pattern = gate_pattern.copy()
    zero_gate_pattern[:] = 0
    for to_layer_name in layer_names:
        for from_layer_name in layer_names:
            gate_pattern[:] = 0
            gate_pattern[vm.network.get_gate_index(to_layer_name,from_layer_name)] = omega
            vm.train(
                {from_layer_name:'pattern', 'A':gate_pattern},
                {to_layer_name:'pattern','A':zero_gate_pattern})
    # set value to_layer_name
    for to_layer_name in layer_names:
        gate_pattern[:] = 0
        gate_pattern[vm.network.get_gate_index(to_layer_name,'OP1')] = omega
        vm.train({'OPC':vm.encode('set'),'OP2':vm.encode(to_layer_name)},{'A':gate_pattern})
        vm.train({'OPC':vm.encode('set'),'OP2':vm.encode(to_layer_name),'A':gate_pattern},
                {'A':zero_gate_pattern,'OPC':vm.encode('_')})
    # ccp from_layer_name to_layer_name condition_layer_name (conditional copy)
    for to_layer_name in layer_names:
        for from_layer_name in layer_names:
            for cond_layer_name in layer_names:
                gate_pattern[:] = 0
                gate_pattern[vm.network.get_gate_index(to_layer_name,from_layer_name)] = omega
                vm.train({
                    'OPC':vm.encode('ccp'),
                    'OP1':vm.encode(from_layer_name),
                    'OP2':vm.encode(to_layer_name),
                    'OP3':vm.encode(cond_layer_name),
                    cond_layer_name:vm.encode('TRUE')},
                    {'A':gate_pattern})
                vm.train({
                    'OPC':vm.encode('ccp'),
                    'OP1':vm.encode(from_layer_name),
                    'OP2':vm.encode(to_layer_name),
                    'OP3':vm.encode(cond_layer_name),
                    cond_layer_name:vm.encode('TRUE'),
                    'A':gate_pattern},
                    {'A':zero_gate_pattern,
                    'OPC':vm.encode('_')})
    # compare circuitry
    vm.train({}, {'CO':vm.encode('FALSE')}) # default FALSE behavior
    vm.train({'C1':'pattern','C2':'pattern'}, {'CO':vm.encode('TRUE')}) # unless equal
    # nand circuitry
    vm.train({}, {'NO':vm.encode('TRUE')}) # default TRUE behavior
    vm.train({'N1':vm.encode('TRUE'),'N2':vm.encode('TRUE')}, {'NO':vm.encode('FALSE')}) # unless both
    # mwr value_layer_name pointer_layer_name (memory write)
    gate_pattern[:] = 0
    key_gate_pattern = gate_pattern.copy()
    value_gate_pattern = gate_pattern.copy()
    assoc_gate_pattern = gate_pattern.copy()
    for pointer_layer_name in layer_names:
        for value_layer_name in layer_names:
            key_gate_pattern[:] = 0
            key_gate_pattern[vm.network.get_gate_index('K',pointer_layer_name)] = omega
            vm.train({
                'OPC':vm.encode('mwr'),
                'OP1':vm.encode(value_layer_name),
                'OP2':vm.encode(pointer_layer_name)},
                {'A':key_gate_pattern})
            value_gate_pattern[:] = 0
            value_gate_pattern[vm.network.get_gate_index('V',value_layer_name)] = omega
            assoc_gate_pattern[:] = 0
            assoc_gate_pattern[vm.network.get_gate_index('V','K')] = omega
            vm.train({
                'OPC':vm.encode('mwr'),
                'OP1':vm.encode(value_layer_name),
                'OP2':vm.encode(pointer_layer_name),
                'A':key_gate_pattern},
                {'A':value_gate_pattern})
            vm.train({
                'OPC':vm.encode('mwr'),
                'OP1':vm.encode(value_layer_name),
                'OP2':vm.encode(pointer_layer_name),
                'A':value_gate_pattern},
                {'W':assoc_gate_pattern})
            vm.train({
                'OPC':vm.encode('mwr'),
                'OP1':vm.encode(value_layer_name),
                'OP2':vm.encode(pointer_layer_name),
                'W':assoc_gate_pattern},
                {'OPC':vm.encode('_'),
                 'W':zero_gate_pattern})

def show_tick(vm):
    period = .1
    for t in range(1):
        print('pre: %s'%vm)
        vm.tick()
        print('post: %s'%vm)
        raw_input('.')
        # time.sleep(period)
    
if __name__ == '__main__':
    mvm = mock_nvm()
    # mvm.set_standard_input('NIL',from_human_readable=True)
    # print(mvm.get_standard_output(to_human_readable=True))
    flash_nrom(mvm)
    # print(mvm.network.transitions['{0}'])
    mvm.show()
    show_tick(mvm)

    # # conditional copies
    # mvm.set_instruction('set','NIL','{0}')
    # show_tick(mvm)
    # show_tick(mvm)
    # show_tick(mvm)
    # mvm.set_instruction('set','TRUE','{1}')
    # show_tick(mvm)
    # show_tick(mvm)
    # show_tick(mvm)
    # mvm.set_instruction('set','FALSE','{2}')
    # show_tick(mvm)
    # show_tick(mvm)
    # show_tick(mvm)
    # raw_input('...')
    # mvm.set_instruction('ccp','{0}','{1}','{2}')
    # show_tick(mvm)
    # mvm.set_instruction('ccp','{0}','{2}','{1}')
    # show_tick(mvm)
    
    # # compare/logic
    # print('set!')
    # mvm.set_instruction('set','TRUE','N1')
    # # mvm.set_instruction('set','TRUE','C1')
    # show_tick(mvm)
    # show_tick(mvm)
    # show_tick(mvm)
    # show_tick(mvm)
    # print('set!')
    # mvm.set_instruction('set','TRUE','N2')
    # # mvm.set_instruction('set','TRUE','C2')
    # show_tick(mvm)
    # show_tick(mvm)
    # show_tick(mvm)
    # show_tick(mvm)
    # print('set!')
    # mvm.set_instruction('set','NIL','N2')
    # # mvm.set_instruction('set','NIL','C2')
    # show_tick(mvm)
    # show_tick(mvm)
    # show_tick(mvm)

    # memory
    print('set!')
    mvm.set_instruction('set','TRUE','{0}')
    show_tick(mvm)
    show_tick(mvm)
    show_tick(mvm)
    show_tick(mvm)
    print('set!')
    mvm.set_instruction('set','NIL','{1}')
    show_tick(mvm)
    show_tick(mvm)
    show_tick(mvm)
    show_tick(mvm)
    print('set!')
    mvm.set_instruction('mwr','{0}','{1}')
    show_tick(mvm)
    show_tick(mvm)
    show_tick(mvm)
    show_tick(mvm)
    show_tick(mvm)
    show_tick(mvm)
    print('set!')
    mvm.set_instruction('set','_','K')
    show_tick(mvm)
    show_tick(mvm)
    show_tick(mvm)
    show_tick(mvm)
    print('set!')
    mvm.set_instruction('set','FALSE','V')
    show_tick(mvm)
    show_tick(mvm)
    show_tick(mvm)
    show_tick(mvm)
    print('set!')
    mvm.set_instruction('set','NIL','K')
    show_tick(mvm)
    show_tick(mvm)
    show_tick(mvm)
    show_tick(mvm)

    
    
    # mvm.hide()
