"""
Symbolically implemented reference machine
"""
from nvm_assembler import preprocess

class RefVM:
    def __init__(self, register_names):
        self.register_names = register_names
        self.programs = {}
        self.active_program = None
        self.layers = {reg: None
            for reg in self.register_names + \
            ["ip","co","mf"]}
        self.memory = {}
        self.stack = []
        self.exit = False

    def assemble(self, program, name, verbose=0):
        lines, labels = preprocess(program, self.register_names)
        self.programs[name] = (lines, labels)

    def load(self, program_name, initial_state):
        # set program pointer
        self.active_program = program_name
        self.layers["ip"] = 0
        self.layers["mf"] = 0
        self.layers["co"] = False
        self.exit = False
        # set initial activities
        self.layers.update(initial_state)

    def state_string(self):
        lines, labels = self.programs[self.active_program]
        return "ip %s: "%self.layers["ip"] + \
            " ".join(lines[self.layers["ip"]]) + ", " + \
            ",".join([
                "%s:%s"%(r,self.layers[r]) for r in self.register_names
            ])

    def decode_state(self):
        return dict(self.layers)

    def at_exit(self):
        return self.exit

    def step(self, verbose=False):
        # load instruction
        lines, labels = self.programs[self.active_program]
        line = lines[self.layers["ip"]]

        # execute instruction
        if line[0] == "exit": self.exit = True
        if line[0] == "movv": self.layers[line[1]] = line[2]
        if line[0] == "movd": self.layers[line[1]] = self.layers[line[2]]
        if line[0] == "cmpv":
            self.layers["co"] = (
                self.layers[line[1]] == line[2])
        if line[0] == "cmpd":
            self.layers["co"] = (
                self.layers[line[1]] == self.layers[line[2]])
        if line[0] == "jie":
            if self.layers["co"]:
                self.layers["ip"] = labels[line[1]]-1
        if line[0] == "jmpv":
            self.layers["ip"] = labels[line[1]]-1
        if line[0] == "jmpd":
            self.layers["ip"] = labels[self.layers[line[1]]]-1

        if line[0] == "nxt": self.layers["mf"] += 1
        if line[0] == "prv": self.layers["mf"] -= 1
        if line[0] == "mem":
            mf = self.layers["mf"]
            if mf not in self.memory: self.memory[mf] = {}
            self.memory[mf][line[1]] = self.layers[line[1]]
        if line[0] == "rem":
            mf = self.layers["mf"]
            self.layers[line[1]] = self.memory[mf][line[1]]

        if line[0] == "subv":
            self.stack.append(self.layers["ip"])
            self.layers["ip"] = labels[line[1]]-1
        if line[0] == "subd":
            self.stack.append(self.layers["ip"])
            self.layers["ip"] = labels[self.layers[line[1]]]-1
        if line[0] == "ret":
            self.layers["ip"] = self.stack.pop()

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
