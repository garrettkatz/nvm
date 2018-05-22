import itertools as it
import unittest as ut
from refvm import RefVM

class RefVMTestCase(ut.TestCase):

    def _test(self, program, trace, num_registers=1, verbose=0):
    
        rvm = RefVM(["r%d"%r for r in range(num_registers)])
        rvm.assemble(program, "test")
        rvm.load("test", trace[0])
        lines, labels = rvm.programs["test"]

        if verbose > 0: print()
        for t in it.count(0):
            if verbose > 0:
                print(t,rvm.state_string())
                print({r:rvm.registers[r] for r in rvm.register_names})
                print(trace[t])
            self.assertTrue(
                {r:rvm.registers[r] for r in rvm.register_names} == trace[t])
            if lines[rvm.registers["ip"]][0] == "exit": break
            rvm.step()

    def test_movv(self):

        program = """
        start:  mov r0 A
                exit
        """
        trace = [
            {"r0": "B"},
            {"r0": "A"}]

        self._test(program, trace, num_registers=1)

    def test_movd(self):

        program = """
        start:  mov r0 A
                mov r1 r0
                exit
        """
        trace = [
            {"r0": None, "r1": None},
            {"r0": "A", "r1": None},
            {"r0": "A", "r1": "A"}]

        self._test(program, trace, num_registers=2)

    def test_jmpv(self):

        program = """
        start:  jmp end
                mov r0 A
        end:    exit
        """
        trace = [
            {"r0": None},
            {"r0": None}]

        self._test(program, trace, num_registers=1, verbose=0)

    def test_jmpd(self):

        program = """
        start:  mov r0 end
                jmp r0
                mov r0 A
        end:    exit
        """
        trace = [
            {"r0": None},
            {"r0": "end"},
            {"r0": "end"},]

        self._test(program, trace, num_registers=1, verbose=0)

    def test_cmpv(self):

        program = """
        start:  mov r0 A        
                cmp r0 B
                jie eq1
                mov r0 B
        eq1:    mov r0 C
                cmp r0 C
                jie eq2
                mov r0 B
        eq2:    mov r0 A
        end:    exit
        """
        trace = [
            {"r0": None},
            {"r0": "A"},
            {"r0": "A"},
            {"r0": "A"},
            {"r0": "B"},
            {"r0": "C"},
            {"r0": "C"},
            {"r0": "C"},
            {"r0": "A"},]

        self._test(program, trace, num_registers=1, verbose=0)

    def test_cmpd(self):

        program = """
        start:  mov r0 A
                mov r1 B
                cmp r0 r1
                jie eq1
                mov r0 D
        eq1:    mov r0 C
                mov r1 C
                cmp r0 r1
                jie eq2
                mov r0 B
        eq2:    mov r0 A
        end:    exit
        """
        trace = [
            {"r0": None, "r1": None}, # start
            {"r0": "A", "r1": None},
            {"r0": "A", "r1": "B"},
            {"r0": "A", "r1": "B"},
            {"r0": "A", "r1": "B"},
            {"r0": "D", "r1": "B"}, # eq1
            {"r0": "C", "r1": "B"},
            {"r0": "C", "r1": "C"},
            {"r0": "C", "r1": "C"},
            {"r0": "C", "r1": "C"},
            {"r0": "A", "r1": "C"},]

        self._test(program, trace, num_registers=2, verbose=0)

    def test_memory(self):

        program = """
        start:  mov r0 A
                mem r0
                nxt
                mov r0 B
                prv
                rem r0
        overw:  mov r0 C
                mem r0
                nxt
                mov r0 D
                prv
                rem r0
        end:    exit
        """
        trace = [
            {"r0": None}, # start
            {"r0": "A"},
            {"r0": "A"},
            {"r0": "A"},
            {"r0": "B"},
            {"r0": "B"},
            {"r0": "A"}, # overw
            {"r0": "C"},
            {"r0": "C"},
            {"r0": "C"},
            {"r0": "D"},
            {"r0": "D"},
            {"r0": "C"},
            ]

        self._test(program, trace, num_registers=1, verbose=0)

    def test_subv(self):

        program = """
        start:  sub foo
                exit

        foo:    mov r0 A
                sub bar
                mov r0 B
                sub bar
                mov r0 C
                ret

        bar:    mov r0 D
                ret
        """
        trace = [
            {"r0": None}, # start
            {"r0": None}, # foo
            {"r0": "A"},
            {"r0": "A"}, # bar
            {"r0": "D"},
            {"r0": "D"},
            {"r0": "B"},
            {"r0": "B"},
            {"r0": "D"},
            {"r0": "D"},
            {"r0": "C"},
            {"r0": "C"},
            {"r0": "C"},
            ]

        self._test(program, trace, num_registers=1, verbose=0)

    def test_subd(self):

        program = """
        start:  sub r1
                exit

        foo:    mov r0 A
                mov r1 bar
                sub r1
                mov r0 B
                sub r1
                mov r0 C
                ret

        bar:    mov r0 D
                ret
        """
        trace = [
            {"r0": None, "r1": "foo"}, # start
            {"r0": None, "r1": "foo"}, # foo
            {"r0": "A", "r1": "foo"},
            {"r0": "A", "r1": "bar"},
            {"r0": "A", "r1": "bar"}, # bar
            {"r0": "D", "r1": "bar"},
            {"r0": "D", "r1": "bar"},
            {"r0": "B", "r1": "bar"},
            {"r0": "B", "r1": "bar"},
            {"r0": "D", "r1": "bar"},
            {"r0": "D", "r1": "bar"},
            {"r0": "C", "r1": "bar"},
            {"r0": "C", "r1": "bar"},
            {"r0": "C", "r1": "bar"},
            ]

        self._test(program, trace, num_registers=2, verbose=0)

if __name__ == "__main__":
    test_suite = ut.TestLoader().loadTestsFromTestCase(RefVMTestCase)
    ut.TextTestRunner(verbosity=2).run(test_suite)
