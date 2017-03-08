import sys, time
import multiprocessing as mp
import numpy as np
import scipy as sp
import visualizer as vz
import mock_net as mn

constants = ['TRUE','FALSE','NIL','_']

class NVM:
    def __init__(self, coding, network):
        self.coding = coding
        self.network = network
        self.visualizing = False
        # Encode layer names and constants
        for symbol in self.network.get_layer_names() + constants:
            self.coding.encode(symbol)
    def tick(self):
        # answer any visualizer request
        if self.visualizing:
            if self.viz_pipe.poll():
                # flush request
                self.viz_pipe.recv()
                # respond with data
                self.send_viz_data()
        # network update
        self.network.tick()
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
            self.viz_pipe.send(self.coding.decode(pattern)) # value
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
            pattern = self.coding.encode(message)
        else:
            pattern = np.fromstring(pattern,dtype=float)
        self.network.set_patterns([('STDI', pattern)])
    def get_output(self, io_module_name, to_human_readable=True):
        pattern = self.network.get_pattern('STDO')
        if to_human_readable:
            message = self.coding.decode(pattern)
        else:
            message = pattern.tobytes()
        return message
    def set_instruction(self, opcode, *operands):
        # set operation
        pattern_list = [('OPC',self.coding.encode(opcode))]
        for op in range(len(operands)):
            pattern_list.append(('OP%d'%(op+1), self.coding.encode(operands[op])))
        self.network.set_patterns(pattern_list)
        # clear gates
        pattern_list.append(('V',self.network.get_pattern('V')*0))
    def learn(self, module_name, pattern_list, next_pattern_list):
        # train module with module.learn
        self.network.get_module(module_name).learn(pattern_list, next_pattern_list)
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

def flash(vm):
    # train vm on instruction set
    omega = np.tanh(1)
    gate_index = vm.network.modules['gating'].gate_index
    gate_pattern = np.zeros(vm.network.modules['gating'].layer_size)
    # set value to_register
    set_pattern = vm.coding.encode('set')
    for to_register_name in vm.network.register_names:
        to_register_pattern = vm.coding.encode(to_register_name)
        gate_pattern[:] = 0
        gate_pattern[gate_index[to_register_name,'OP1']] = omega
        vm.learn('gating',
            [('OPC',set_pattern),('OP2',to_register_pattern)],
            [('V',gate_pattern)])    
    # copy from_register to_register
    copy_pattern = vm.coding.encode('copy')
    for to_register_name in vm.network.register_names:
        to_register_pattern = vm.coding.encode(to_register_name)
        for from_register_name in vm.network.register_names:
            from_register_pattern = vm.coding.encode(from_register_name)
            gate_pattern[:] = 0
            gate_pattern[gate_index[to_register_name,from_register_name]] = omega
            vm.learn('gating',
                [('OPC',copy_pattern),('OP1',from_register_pattern),('OP2',to_register_pattern)],
                [('V',gate_pattern)])
        # if operation == self.machine_readable['get']: # device_name, register
        #     # gated NN behaviors:
        #     # copy layer
        #     self.registers[operands[1]] = self.devices[operands[0]].output_layer
        # if operation == self.machine_readable['put']: # device_name, register
        #     # gated NN behaviors:
        #     # copy layer
        #     self.devices[operands[0]].input_layer = self.registers[operands[1]]

def show_tick(vm):
    period = 1
    for t in range(4):
        vm.tick()
        time.sleep(period)
    
if __name__ == '__main__':
    mvm = mock_nvm()
    mvm.set_input('NIL','stdio',from_human_readable=True)
    # print(mvm.get_output('stdio',to_human_readable=True))
    flash(mvm)
    mvm.show()
    mvm.set_instruction('set','NIL','{0}')
    show_tick(mvm)
    mvm.set_instruction('set','TRUE','{1}')
    show_tick(mvm)
    mvm.set_instruction('set','FALSE','{2}')
    show_tick(mvm)
    mvm.set_instruction('copy','{1}','{2}')
    show_tick(mvm)

    
    
    # mvm.hide()
