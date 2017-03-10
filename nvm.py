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
    def set_input(self, message, io_module_name, from_human_readable=True):
        if from_human_readable:
            pattern = self.encode(message)
        else:
            pattern = np.fromstring(pattern,dtype=float)
        self.network.set_pattern('STDI', pattern)
    def get_output(self, io_module_name, to_human_readable=True):
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
    def train(self, module_name, pattern_list, next_pattern_list):
        # train module with module.train
        self.network.get_module(module_name).train(pattern_list, next_pattern_list)
    def quit(self):
        self.hide()
        sys.exit(0)

def mock_nvm(num_registers=3, layer_size=32):
    coding = mn.MockCoding(layer_size)
    stdio = mn.MockIOModule(module_name='stdio', layer_size=layer_size)
    net = mn.MockNet(num_registers, layer_size, io_modules=[stdio])
    return NVM(coding, net)

def run_viz(nvm_pipe):
    viz = vz.Visualizer(nvm_pipe)
    viz.launch()

def flash_nrom(vm):
    # train vm on instruction set
    omega = np.tanh(1)
    gate_index_map = vm.network.get_module('gating').gate_index_map
    gate_pattern = np.zeros(vm.network.get_module('gating').layer_size)
    # get non-gate layer names
    layer_names = vm.network.get_layer_names()
    for layer_name in vm.network.get_module('gating').layer_names:
        layer_names.remove(layer_name)
    # set value to_layer_name
    for to_layer_name in layer_names:
        gate_pattern[:] = 0
        gate_pattern[gate_index_map[to_layer_name,'OP1']] = omega
        vm.train('gating',
            [('OPC',vm.encode('set')),('OP2',vm.encode(to_layer_name))],
            [('A',gate_pattern)])    
    # ccp from_layer_name to_layer_name condition_layer_name (conditional copy)
    for to_layer_name in layer_names:
        for from_layer_name in layer_names:
            for cond_layer_name in layer_names:
                gate_pattern[:] = 0
                gate_pattern[gate_index_map[to_layer_name,from_layer_name]] = omega
                vm.train('gating',
                    [('OPC',vm.encode('ccp')),
                     ('OP1',vm.encode(from_layer_name)),
                     ('OP2',vm.encode(to_layer_name)),
                     ('OP3',vm.encode(cond_layer_name)),
                     (cond_layer_name, vm.encode('TRUE'))],
                    [('A',gate_pattern)])
    # compare circuitry
    vm.train('compare', [], [('CO',vm.encode('FALSE'))]) # default FALSE behavior
    vm.train('compare', [('C1','pattern'),('C2','pattern')], [('CO',vm.encode('TRUE'))]) # unless equal
    # nand circuitry
    vm.train('nand', [], [('NO',vm.encode('TRUE'))]) # default TRUE behavior
    vm.train('nand',
        [('N1',vm.encode('TRUE')),('N2',vm.encode('TRUE'))],
        [('NO',vm.encode('FALSE'))]) # unless both

def show_tick(vm):
    period = .1
    for t in range(1):
        vm.tick()
        pattern_list = vm.network.list_patterns()
        vmstr = ''
        for (layer_name, pattern) in pattern_list:
            if vm.decode(pattern)=='<?>': continue
            vmstr += '%s:%s;'%(layer_name, vm.decode(pattern))
        print(vmstr)
        raw_input('.')
        # time.sleep(period)
    
if __name__ == '__main__':
    mvm = mock_nvm()
    mvm.set_input('NIL','stdio',from_human_readable=True)
    # print(mvm.get_output('stdio',to_human_readable=True))
    flash_nrom(mvm)
    mvm.show()
    show_tick(mvm)

    # # conditional copies
    # mvm.set_instruction('set','NIL','{0}')
    # show_tick(mvm)
    # mvm.set_instruction('set','TRUE','{1}')
    # show_tick(mvm)
    # mvm.set_instruction('set','FALSE','{2}')
    # show_tick(mvm)
    # raw_input('...')
    # mvm.set_instruction('ccp','{0}','{1}','{2}')
    # show_tick(mvm)
    # mvm.set_instruction('ccp','{0}','{2}','{1}')
    # show_tick(mvm)
    
    # compare/logic
    print('set!')
    mvm.set_instruction('set','TRUE','N1')
    show_tick(mvm)
    show_tick(mvm)
    show_tick(mvm)
    show_tick(mvm)
    print('set!')
    mvm.set_instruction('set','TRUE','N2')
    show_tick(mvm)
    show_tick(mvm)
    show_tick(mvm)
    show_tick(mvm)
    print('set!')
    mvm.set_instruction('set','NIL','N2')
    show_tick(mvm)
    show_tick(mvm)
    show_tick(mvm)

    
    
    # mvm.hide()
