"""
NVM standard library routines for logic and arithmetic
r0, r1, ... are replaced with register names in order
"""

##### Boolean operations (not, and, or)
# r0 and r1 should be names of two registers used as operands
def logic_routines(r0, r1):
    return """
                exit

# logical-not of r0
# overwrites r0 with the result
stl.not:        cmp {r0} true
                jie stl.not.false
                mov {r0} true
                ret
stl.not.false:  mov {r0} false
                ret
    
# logical-and of r0 and r1
# overwrites r0 with result
stl.and:        cmp {r0} false
                jie stl.and.false
                cmp {r1} false
                jie stl.and.false
                mov {r0} true
                ret
stl.and.false:  mov {r0} false
                ret

# logical-or of r0 and r1
# overwrites r0 with result
stl.or:         cmp {r0} true
                jie stl.or.true
                cmp {r1} true
                jie stl.or.true
                mov {r0} false
                ret
stl.or.true:    mov {r0} true
                ret

""".format(r0=r0, r1=r1)

##### Contiguous array memory management
# r0 is pointer register and r1 is value register
def array_routines(r0, r1):
    return """
                exit

stl.or.true:    mov {r0} true
                ret

""".format(r0=r0, r1=r1)

if __name__ == "__main__":
    
    print(logic_routines("ra", "rb"))
