register_names = ["r0", "r1"]

programs = {
"myfirstprogram":"""

### computes logical-and of r0 and r1, overwriting r0 with result

        nop           # do nothing
        sub and       # call logical-and sub-routine
        exit          # halt execution

and:    cmp r0 false  # compare first conjunct to false
        jie and.f     # jump, if equal to false, to and.f label
        cmp r1 false  # compare second conjunct to false
        jie and.f     # jump, if equal false, to and.f label
        mov r0 true   # both conjuncts true, set r0 to true
        ret           # return from sub-routine
and.f:  mov r0 false  # a conjunct was false, set r0 to false
        ret           # return from sub-routine

"""}

from nvm.nvm import make_scaled_nvm

my_nvm = make_scaled_nvm(
    register_names = register_names,
    programs = programs,
    orthogonal=True)

my_nvm.assemble(programs)
my_nvm.load("myfirstprogram",
    initial_state = {"r0":"true","r1":"false"})

print(my_nvm.net.layers["r0"].shape)
print(my_nvm.net.activity["r0"].T)

v = my_nvm.net.activity["r0"].T
print(my_nvm.net.layers["r0"].coder.decode(v))

print(my_nvm.decode_state(layer_names=register_names))

import itertools

for t in itertools.count():
    my_nvm.net.tick()
    if my_nvm.at_exit(): break

print(my_nvm.net.layers["opc"].coder.decode(my_nvm.net.activity["opc"]) == "exit")

print(t)
print(my_nvm.decode_state(layer_names=register_names))
