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

if __name__ == "__main__":
    test_suite = ut.TestLoader().loadTestsFromTestCase(RefVMTestCase)
    ut.TextTestRunner(verbosity=2).run(test_suite)
