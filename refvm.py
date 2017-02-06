import multiprocessing as mp
import numpy as np
import vm_gui

layer_size = 32

def bits(i):
    b = np.empty((layer_size,),dtype=np.uint8)
    for j in range(layer_size):
        b[j] = 255*(1 & (i >> j))
    return tuple(b)

class RefVM:
    """
    """
    def __init__(self, register_names=['{'+str(r)+'}' for r in range(8)]):
        # setup literals
        self.literals = []
        self.human_readable = {} # character strings
        self.machine_readable = {} # bit strings
        literals = [
            'nop','set','copy','store','prepend','read','write','next','compare','jump','nor', 'get', 'put'] # instructions
        literals += register_names # registers
        literals += ['FALSE','TRUE'] # booleans
        literals += ['NIL','_'] # placeholders
        for literal in literals:
            self.add_literal(literal)
        self.viz_on = False
        # setup machine state
        self.instruction_pointer = None
        self.instruction = [self.machine_readable['_']]*4
        self.memory = {} # memory[pointer] = (value, next_pointer)
        self.free_key = 0
        self.instruction_size = 4
        self.registers = {self.machine_readable[r]:self.machine_readable['_'] for r in register_names}
        self.devices = {}
        # init with nop program
        assembly_code = "nop"
        object_code, label_table = self.assemble(assembly_code)
        self.load(object_code, label_table)
    def add_literal(self, literal):
        if literal not in self.literals:
            ell = -len(self.machine_readable)
            # ell = tuple(np.tanh(np.random.randn(layer_size)))
            self.human_readable[ell] = literal
            self.machine_readable[literal] = ell
            self.literals.append(literal)
    def hr(self, mr):
        if mr in self.human_readable:
            # return '%s|'%mr + self.human_readable[mr]
            return self.human_readable[mr]
        else:
            return mr
    def install_device(self, device_name, device):
        self.add_literal(device_name)
        self.devices[self.machine_readable[device_name]] = device # device should implement put(), get()
    def disp(self, mr=False):
        print('ip: %s'%(self.instruction_pointer,))
        print([self.hr(op) for op in self.instruction])
        if mr:
            print(['%s:%s'%(self.registers[r],self.hr(self.registers[r])) for r in sorted(self.registers.keys(),reverse=True)])
        else:
            print([self.hr(self.registers[r]) for r in sorted(self.registers.keys(),reverse=True)])
        keys = sorted(self.memory.keys())
        # print([(k, self.memory[k]) for k in keys])
        print([self.hr(self._get_head(k)) for k in keys])
        print(self.devices)
        print(self.machine_readable)
    def _get_tail(self, pointer):
        return self.memory[pointer][1]
    def _get_head(self, pointer):
        return self.memory[pointer][0]
    def _set_head(self, pointer, value):
        self.memory[pointer] = (value, self._get_tail(pointer))
    def _cons(self, head, tail=None):
        pointer = self.free_key
        # pointer = tuple(np.tanh(np.random.randn(layer_size)))
        if tail is None: tail = pointer
        self.memory[pointer] = (head, tail)
        self.free_key += 1
        return pointer
    def assemble(self, assembly_code):
        """
        assembler:
          preprocess assembly (comments, whitespace)
          save label offsets
          output: [[op,op,op,label],...], {label:index} ("object" code)
        loader:
          load the program into memory, recording the nvm-determined keys
          traverse program in memory and overwrite jump labels with nvm-determined keys
          no output, but k(op, k(op, k(op, k(k, ...))) in memory with labels updated to nvm-determined keys

        assembly_code: string of line separated instructions
        return object_code: [[op, op, op, label], ...], {label:index}
        """
        object_code = []
        label_table = {}
        offset = 0
        for line in assembly_code.split("\n"):
            # remove comments
            comment_index = line.find("#")
            if comment_index < 0: comment_index = len(line)
            line = line[:comment_index]
            # remove labels
            label_index = line.find(":")
            instruction = line[label_index+1:]
            # strip whitespace
            instruction = instruction.strip()
            # skip empty lines
            if instruction == '': continue
            # store label
            if label_index > -1:
                label = line[:label_index]
                label_table[label] = offset
            # store operands
            operands = instruction.split(' ')
            for op in range(len(operands), self.instruction_size):
                operands.append('_')
            object_code.append(operands)
            offset += 1
        # convert literals to machine-readable
        for instruction in object_code:
            for op in range(self.instruction_size):
                self.add_literal(instruction[op])
                instruction[op] = self.machine_readable[instruction[op]]
        for label in label_table:
            self.add_literal(label)
        label_table = {self.machine_readable[label]:label_table[label] for label in label_table}
        # done
        return object_code, label_table
    def load(self, object_code, label_table):
        # load the program into memory, recording the nvm-determined keys
        key = None
        keys = []
        for instruction in object_code[::-1]:
            for op in instruction[::-1]:
                key = self._cons(op, key)
            keys.append(key)
        keys.reverse()
        # traverse program in memory and overwrite jump labels with nvm-determined keys
        while key != self._get_tail(key):
            op = self._get_head(key)
            if op in label_table:
                self._set_head(key, keys[label_table[op]])
            key = self._get_tail(key)
        # initialize instruction pointer
        self.instruction_pointer = keys[0]
    def show_gui(self):
        if not self.viz_on:
            self.viz_on = True
            self.vm_pipe_to_gui, gui_pipe_to_vm = mp.Pipe()
            self.gui_process = mp.Process(target=_run_gui, args=(gui_pipe_to_vm,))
            self.gui_process.start()
    def hide_gui(self):
        if self.viz_on:
            self.vm_pipe_to_gui.send('q')
            self.gui_process.join()
            self.viz_on = False
    def gui_state(self):
        layers = [('{ip}', self.hr(self.instruction_pointer), tuple(bits(self.instruction_pointer)))]
        for op in range(self.instruction_size):
            layers.append(('{i%d}'%op, self.hr(self.instruction[op]), tuple(bits(self.instruction[op]))))
        for r in sorted(self.registers.keys(),reverse=True):
            layers.append((self.hr(r), self.hr(self.registers[r]), tuple(bits(self.registers[r]))))
        return tuple(layers)
    def tick(self):
        # handle visualization
        if self.viz_on:
            if self.vm_pipe_to_gui.poll():
                msg = self.vm_pipe_to_gui.recv()
                state = self.gui_state()
                self.vm_pipe_to_gui.send(state)
        # execute instruction
        self.instruction = []
        for op in range(4):
            # gated NN behaviors:
            # attractor activity
            # hetero-associative activity
            # copy layer
            self.instruction.append(self._get_head(self.instruction_pointer))
            self.instruction_pointer = self._get_tail(self.instruction_pointer)
        operation, operands = self.instruction[0], self.instruction[1:]
        if operation == self.machine_readable['nop']:
            pass
        if operation == self.machine_readable['set']: # value, register
            # gated NN behaviors:
            # copy layer
            self.registers[operands[1]] = operands[0]
        if operation == self.machine_readable['copy']: # source register, target register
            # gated NN behaviors:
            # copy layer
            self.registers[operands[1]] = self.registers[operands[0]]
        if operation == self.machine_readable['store']: # value register, pointer register
            # gated NN behaviors:
            # unused waypoint activation
            # fast attractor learning
            # fast hetero-associative learning
            # copy layer
            self.registers[operands[1]] = self._cons(self.registers[operands[0]])
        if operation == self.machine_readable['prepend']: # value register, pointer register
            # gated NN behaviors:
            # unused waypoint activation
            # fast attractor learning
            # fast hetero-associative learning
            # copy layer
            self.registers[operands[1]] = self._cons(self.registers[operands[0]], self.registers[operands[1]])
        if operation == self.machine_readable['read']: # value register, pointer register
            # gated NN behaviors:
            # attractor activity
            # hetero-associative activity
            # copy layer
            self.registers[operands[0]] = self._get_head(self.registers[operands[1]])
        if operation == self.machine_readable['write']: # value register, pointer register
            # gated NN behaviors:
            # fast hetero-associative learning
            # copy layer
            self._set_head(self.registers[operands[1]], self.registers[operands[0]])
        if operation == self.machine_readable['next']: # pointer register
            # gated NN behaviors:
            # attractor activity
            # copy layer
            self.registers[operands[0]] = self._get_tail(self.registers[operands[0]])
        if operation == self.machine_readable['compare']: # value register, value register, result register
            # gated NN behaviors:
            # comparison circuit            
            # copy layer
            if self.registers[operands[0]] == self.registers[operands[1]]:
                self.registers[operands[2]] = self.machine_readable['TRUE']
            else:
                self.registers[operands[2]] = self.machine_readable['FALSE']
        if operation == self.machine_readable['jump']: # condition register, instruction pointer
            # gated NN behaviors:
            # conditional copy layer
            if self.registers[operands[0]] != self.machine_readable['FALSE']:
                self.instruction_pointer = operands[1]
        if operation == self.machine_readable['nor']: # disjunct register, disjunct register, result register
            # gated NN behaviors:
            # nor circuit
            # copy layer
            disjuncts = [(self.registers[op] != self.machine_readable['FALSE']) for op in operands[:2]]
            if not (disjuncts[1] or disjuncts[0]):
                self.registers[operands[2]] = self.machine_readable['TRUE']
            else:
                self.registers[operands[2]] = self.machine_readable['FALSE']
        if operation == self.machine_readable['get']: # device_name, register
            # gated NN behaviors:
            # copy layer
            self.registers[operands[1]] = self.devices[operands[0]].output_layer
        if operation == self.machine_readable['put']: # device_name, register
            # gated NN behaviors:
            # copy layer
            self.devices[operands[0]].input_layer = self.registers[operands[1]]

def _run_gui(gui_pipe_to_vm):
    vmg = vm_gui.VMGui(gui_pipe_to_vm, history = 512)


class RefIODevice:
    def __init__(self, machine_readable, human_readable):
        self.machine_readable = machine_readable
        self.human_readable = human_readable
        self.input_layer = self.machine_readable['_']
        self.output_layer = self.machine_readable['_']
    def put(self, hr):
        self.output_layer = self.machine_readable[hr]
    def peek(self):
        return self.human_readable[self.input_layer]

test_assembly_code = """
# io
set TRUE {1}
loop: get rvmio {0}
put rvmio {0}
compare {0} {1} {2}
jump {2} loop
put rvmio {1}
# build the list
set a {0}
store {0} {1}
set b {0}
prepend {0} {1}
set c {0}
skip: prepend {0} {1}
# read the list
copy {1} {2}
read {0} {1}
next {1}
read {0} {1}
next {1}
read {0} {1}
# overwrite the list
copy {2} {1}
next {1}
set z {0}
write {0} {1}
# control flow
set a {0}
set b {1}
compare {0} {1} {3}
compare {0} {1} {4}
nor {3} {4} {5}
jump {5} lab
set z {0} # dead code?
lab: set z {1}
# end
nop
"""

if __name__ == '__main__':
    rvm = RefVM()
    rvmio = RefIODevice(rvm.machine_readable, rvm.human_readable)
    rvmio.put('TRUE')
    rvm.install_device('rvmio',rvmio)
    assembly_code = test_assembly_code
    object_code, label_table = rvm.assemble(assembly_code)
    print(object_code)
    print(label_table)
    rvm.load(object_code, label_table)
    rvm.disp()
    for t in range(30):
        rvm.tick()
        print('\nt=%d:'%t)
        rvm.disp()
        # if rvm.instruction_pointer == rvm._get_tail(rvm.instruction_pointer): break
        raw_input('')
    print(rvmio.input_layer)
