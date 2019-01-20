programs = {
"myfirstprogram":"""

        nop           # do nothing
        mov r0 true   # set r0 to true
        mov r1 false  # set r1 to false
        sub and       # call and sub-routine
        exit          # halt execution

and:    cmp r0 false  # compare first conjunct to false
        jie and.f     # jump if equal to and.f label
        cmp r1 false  # compare second conjunct to false
        jie and.f     # jump if equal to and.f label
        mov r0 true   # both conjuncts true, set r0 to true
        ret           # return from sub-routine
and.f:  mov r0 false  # a conjunct was false, set r0 to false
        ret           # return from sub-routine

"""}

from nvm.nvm import make_scaled_nvm

my_nvm = make_scaled_nvm(
    register_names = ["r0", "r1"],
    programs = programs,
    orthogonal=True)

my_nvm.assemble(programs)
my_nvm.load("myfirstprogram",
    initial_state = {"r0":"false","r1":"true"})

print(my_nvm.net.activity["r0"].T)

v = my_nvm.net.activity["r0"].T
print(my_nvm.net.layers["r0"].coder.decode(v))

print(my_nvm.decode_state(layer_names=["r0","r1"]))

import itertools as it

for t in it.count():
    my_nvm.net.tick()
    if my_nvm.at_exit(): break

print(t)
print(my_nvm.decode_state(layer_names=["r0","r1"]))
