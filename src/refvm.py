"""
Symbolically implemented reference machine
"""
from nvm_assembler import preprocess

class RefVM:
    def __init__(self, register_names):
        self.register_names = register_names
        self.programs = {}
        self.active_program = None
        self.registers = {reg: None
            for reg in self.register_names + \
            ["ip","co","mf"]}
        self.memory = {}
        self.stack = []

    def assemble(self, program, name):
        lines, labels = preprocess(program, self.register_names)
        self.programs[name] = (lines, labels)

    def load(self, program_name, initial_state):
        # set program pointer
        self.active_program = program_name
        self.registers["ip"] = 0
        self.registers["mf"] = 0
        self.registers["co"] = False
        # set initial activities
        self.registers.update(initial_state)

    def state_string(self):
        lines, labels = self.programs[self.active_program]
        return "ip %s: "%self.registers["ip"] + \
            " ".join(lines[self.registers["ip"]]) + ", " + \
            ",".join([
                "%s:%s"%(r,self.registers[r]) for r in self.register_names
            ])

    def step(self):
        # load instruction
        lines, labels = self.programs[self.active_program]
        line = lines[self.registers["ip"]]

        # execute instruction
        if line[0] == "movv": self.registers[line[1]] = line[2]
        if line[0] == "movd": self.registers[line[1]] = self.registers[line[2]]
        if line[0] == "cmpv":
            self.registers["co"] = (
                self.registers[line[1]] == line[2])
        if line[0] == "cmpd":
            self.registers["co"] = (
                self.registers[line[1]] == self.registers[line[2]])
        if line[0] == "jie":
            if self.registers["co"]:
                self.registers["ip"] = labels[line[1]]-1
        if line[0] == "jmpv":
            self.registers["ip"] = labels[line[1]]-1
        if line[0] == "jmpd":
            self.registers["ip"] = labels[self.registers[line[1]]]-1

        if line[0] == "nxt": self.registers["mf"] += 1
        if line[0] == "prv": self.registers["mf"] -= 1
        if line[0] == "mem":
            mf = self.registers["mf"]
            if mf not in self.memory: self.memory[mf] = {}
            self.memory[mf][line[1]] = self.registers[line[1]]
        if line[0] == "rem":
            mf = self.registers["mf"]
            self.registers[line[1]] = self.memory[mf][line[1]]

        if line[0] == "subv":
            self.stack.append(self.registers["ip"])
            self.registers["ip"] = labels[line[1]]-1
        if line[0] == "subd":
            self.stack.append(self.registers["ip"])
            self.registers["ip"] = labels[self.registers[line[1]]]-1
        if line[0] == "ret":
            self.registers["ip"] = self.stack.pop()

        # advance instruction pointer
        self.registers["ip"] += 1

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
