"""
Symbolically implemented reference machine
"""
from nvm_assembler import preprocess

class RefVM:
    def __init__(self, register_names):
        self.register_names = register_names
        self.programs = {}
        self.labels = {}
        self.active_program = None
        self.layers = {reg: None
            for reg in self.register_names + \
            ["ip","opc","op1","op2","co","mf"]}
        self.memory = {}
        self.pointers = {}
        self.mpointers = {}
        self.stack = []
        self.exit = False
        self.error = None

    def assemble(self, programs, verbose=0, other_tokens=[]):
        lines, labels, _ = preprocess(programs, self.register_names)

        for name in programs:
            self.programs[name] = (lines[name], labels[name])
            self.labels.update({
                label: (name, l) for label, l in labels[name].items()})

    def load(self, program_name, initial_state):
        # set program pointer
        self.active_program = program_name
        self.layers["ip"] = 0
        self.layers["opc"] = "null"
        self.layers["op1"] = "null"
        self.layers["op2"] = "null"
        self.layers["mf"] = "0"
        self.layers["co"] = False
        self.exit = False
        # set initial activities
        self.layers.update(initial_state)

    def initialize_memory(self, pointers, values):
        for loc in pointers:
            for reg, tok in pointers[loc].items():
                if reg not in self.pointers: self.pointers[reg] = {}
                self.pointers[reg][tok] = loc
        for loc in values:
            for reg, tok in values[loc].items():
                if loc not in self.memory: self.memory[loc] = {}
                self.memory[loc][reg] = tok

    def state_string(self):
        lines, labels = self.programs[self.active_program]
        ip = self.layers["ip"]
        for label, l in labels.items():
            if ip == l:
                ip = label
                break
        return "ip %s: "%ip + \
            " ".join(lines[self.layers["ip"]]) + ", " + \
            ",".join([
                "%s:%s"%(r,self.layers[r]) for r in self.register_names
            ])

    def decode_state(self, layer_names=None):
        if layer_names is None: layer_names = self.layers.keys()
        decoded = {name: self.layers[name] for name in layer_names}
        decoded['co'] = 'true' if decoded['co'] else 'false'
        return decoded

    def at_exit(self):
        # lines, labels = self.programs[self.active_program]
        # line = lines[self.layers["ip"]]
        # return line[0] == "exit"
        return self.exit

    def step(self, verbose=False, max_ticks=0):
        # load instruction
        lines, labels = self.programs[self.active_program]
        line = lines[self.layers["ip"]]
        opc, op1, op2 = line
        self.layers["opc"], self.layers["op1"], self.layers["op2"] = opc, op1, op2

        # execute instruction
        if opc == "exit": self.exit = True
        if opc == "movv": self.layers[op1] = op2
        if opc == "movd": self.layers[op1] = self.layers[op2]
        if opc == "cmpv":
            self.layers["co"] = (
                self.layers[op1] == op2)
        if opc == "cmpd":
            self.layers["co"] = (
                self.layers[op1] == self.layers[op2])
        if opc == "jie":
            if op1 not in self.labels:
                self.error = "jie to non-existent label"
                self.exit = True
            elif self.layers["co"]:
                name, l = self.labels[op1]
                self.active_program, self.layers["ip"] = name, l-1
        if opc == "jmpv":
            if op1 not in self.labels:
                self.error = "jmpv to non-existent label"
                self.exit = True
            else:
                name, l = self.labels[op1]
                self.active_program, self.layers["ip"] = name, l-1
        if opc == "jmpd":
            if self.layers[op1] not in self.labels:
                self.error = "jmpd to non-existent label"
                self.exit = True
            else:
                name, l = self.labels[self.layers[op1]]
                self.active_program, self.layers["ip"] = name, l-1

        if opc == "nxt": self.layers["mf"] = str(int(self.layers["mf"]) + 1)
        if opc == "prv": self.layers["mf"] = str(int(self.layers["mf"]) - 1)
        if opc == "mem":
            mf = self.layers["mf"]
            if mf not in self.memory: self.memory[mf] = {}
            self.memory[mf][op1] = self.layers[op1]
        if opc == "rem":
            mf = self.layers["mf"]
            if mf not in self.memory or op1 not in self.memory[mf]:
                self.exit = True
                self.error = "rem non-existent memory"
            else:
                self.layers[op1] = self.memory[mf][op1]
        if opc == "ref":
            reg, mf = op1, self.layers["mf"]
            if reg not in self.pointers: self.pointers[reg] = {}
            self.pointers[reg][self.layers[reg]] = mf
        if opc == "drf":
            reg, val = op1, self.layers[op1]
            if reg in self.pointers and val in self.pointers[reg]:
                self.layers["mf"] = self.pointers[reg][val]
            else:
                self.error = "drf non-existent pointer"
                self.exit = True
        if opc == "mref":
            reg, mf = op1, self.layers["mf"]
            val = self.layers[reg]
            if reg not in self.pointers or val not in self.pointers[reg]:
                self.error = "mref non-existent pointer"
                self.exit = True
            else:
                self.mpointers[self.pointers[reg][val]] = mf
        if opc == "mdrf":
            mf = self.layers["mf"]
            if mf in self.mpointers:
                self.layers["mf"] = self.mpointers[mf]
            else:
                self.error = "mdrf non-existent pointer"
                self.exit = True

        if opc == "subv":
            if op1 not in self.labels:
                self.error = "subv to non-existent label"
                self.exit = True
            else:
                self.stack.append((self.active_program, self.layers["ip"]))
                name, l = self.labels[op1]
                self.active_program, self.layers["ip"] = name, l-1
        if opc == "subd":
            if self.layers[op1] not in self.labels:
                self.error = "subd to non-existent label"
                self.exit = True
            else:
                self.stack.append((self.active_program, self.layers["ip"]))
                name, l = self.labels[self.layers[op1]]
                self.active_program, self.layers["ip"] = name, l-1
        if opc == "ret":
            if len(self.stack) == 0:
                print('empty stack:')
                print(lines)
                print(labels)
                print(self.layers["ip"])                
            self.active_program, self.layers["ip"] = self.stack.pop()

        if self.error is not None:
            print("RVM ERROR: %s"%self.error)
        else:
            # advance instruction pointer
            self.layers["ip"] += 1        

if __name__ == "__main__":

    programs = {"test":"""
    
start:  mov r1 A
        jmpv jump
        nop
jump:   jie end
        mov r2 r1
        nop
end:    exit
    
    """}

    rvm = RefVM(["r%d"%r for r in range(3)])

    for name, program in programs.items():
        rvm.assemble(program, name)

    rvm.load("test",{"r1":"B","r2":"C"})

    lines, labels = rvm.programs[rvm.active_program]
    for t in range(10):
        print(rvm.state_string())
        if lines[rvm.registers["ip"]][0] == "exit": break
        rvm.step()
