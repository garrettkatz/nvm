import itertools as it
import unittest as ut
from refvm import RefVM
from nvm import make_scaled_nvm

class VMTestCase(ut.TestCase):

    def _test(self, programs, names, traces, memory=None, extra_tokens=[], num_registers=1, verbose=0):
        """
        Assemble all programs
        run all programs in list of names
        memory = (pointers, values)
        """

        vm = self._make_vm(num_registers, programs, extra_tokens=extra_tokens, verbose = (verbose > 0))
        if memory is not None: vm.initialize_memory(*memory)
        
        vm.assemble(programs, verbose=verbose, other_tokens=extra_tokens)

        for name, trace in zip(names, traces):
            vm.load(name, trace[0])
    
            if verbose > 0:
                print()
                print(name)

            for t in it.count(0):
                if vm.at_exit(): break
                state = vm.decode_state()
                if verbose > 0:
                    print(t,vm.state_string())
                    print({r: state[r] for r in vm.register_names+["mf"]})
                    print(trace[t])
                self.assertTrue(
                    {r: state[r] for r in vm.register_names} == trace[t])
                vm.step(verbose=verbose)
            
            self.assertTrue(t == len(trace))

    # @ut.skip("")
    def test_noop(self):

        program = """
        start:  nop
                exit
        """
        trace = [
            {"r0": None},
            {"r0": None}]

        self._test({"test":program}, ["test"], [trace], num_registers=1, verbose=0)

    # @ut.skip("")
    def test_movv(self):

        program = """
        start:  mov r0 A
                exit
        """
        trace = [
            {"r0": "B"},
            {"r0": "A"}]

        self._test({"test": program}, ["test"], [trace],
            extra_tokens=["start","A","B"],
            num_registers=1, verbose=0)

    # @ut.skip("")
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

        self._test({"test": program}, ["test"], [trace],
            extra_tokens=["start","A"],
            num_registers=2, verbose=0)

    # @ut.skip("")
    def test_jmpv(self):

        program = """
        start:  jmp end
                mov r0 A
        end:    exit
        """
        trace = [
            {"r0": None},
            {"r0": None}]

        self._test({"test": program}, ["test"], [trace],
            extra_tokens=["start","end","A"],
            num_registers=1, verbose=0)

    # @ut.skip("")
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

        self._test({"test": program}, ["test"], [trace],
            extra_tokens=["start","end","A"],
            num_registers=1, verbose=0)

    # @ut.skip("")
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

        self._test({"test": program}, ["test"], [trace],
            extra_tokens=["start","eq1","eq2","end","A","B","C"],
            num_registers=1, verbose=0)

    # @ut.skip("")
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

        self._test({"test": program}, ["test"], [trace],
            extra_tokens=["start","eq1","eq2","end","A","B","C","D"],
            num_registers=2, verbose=0)

    # @ut.skip("")
    def test_memr(self):

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

        self._test({"test": program}, ["test"], [trace],
            extra_tokens=["start","overw","end","A","B","C","D"],
            num_registers=1, verbose=0)

    # @ut.skip("")
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
            {"r0": None}, # start: sub foo
            {"r0": None}, # mov r0 A
            {"r0": "A"}, # sub bar
            {"r0": "A"}, # mov r0 D
            {"r0": "D"}, # ret
            {"r0": "D"}, # mov r0 B
            {"r0": "B"}, # sub bar
            {"r0": "B"}, # mov r0 D
            {"r0": "D"}, # ret
            {"r0": "D"}, # mov r0 C
            {"r0": "C"}, # ret
            {"r0": "C"}, # exit
            ]

        self._test({"test": program}, ["test"], [trace],
            extra_tokens=["start","foo","bar","A","B","C","D"],
            num_registers=1, verbose=0)

    # @ut.skip("")
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
            {"r0": None, "r1": "foo"}, # start: sub r1
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
            ]

        self._test({"test": program}, ["test"], [trace],
            extra_tokens=["start","foo","bar","A","B","C","D"],
            num_registers=2, verbose=0)

    # @ut.skip("")
    def test_dref(self):

        program = """
        start:  mov r0 A
                mem r0
                mov r1 X
                ref r1
                nxt
                mov r0 B
                mem r0
                drf r1
                rem r0
                exit
        """
        trace = [
            {"r0": None, "r1": None}, # start: mov
            {"r0": "A", "r1": None}, # mem
            {"r0": "A", "r1": None}, # mov
            {"r0": "A", "r1": "X"}, # ref
            {"r0": "A", "r1": "X"}, # nxt
            {"r0": "A", "r1": "X"}, # mov
            {"r0": "B", "r1": "X"}, # mem
            {"r0": "B", "r1": "X"}, # drf
            {"r0": "B", "r1": "X"}, # rem
            {"r0": "A", "r1": "X"}, # exit
            ]

        self._test({"test": program}, ["test"], [trace],
            extra_tokens=["start","A","X","B"],
            num_registers=2, verbose=0)

    # @ut.skip("")
    def test_twop(self):

        programs = {
        "one": """
        start1: mov r0 A
                mov r1 X
                exit
        """,
        "two": """
        start2: mov r0 B
                mov r1 Y
                exit
        """
        }
        traces = [
            [
                {"r0": None, "r1": None}, # start: mov
                {"r0": "A", "r1": None}, # mov
                {"r0": "A", "r1": "X"}, # exit
            ],
            [
                {"r0": None, "r1": None}, # start: mov
                {"r0": "B", "r1": None}, # mov
                {"r0": "B", "r1": "Y"}, # exit
            ]]

        self._test(programs, ["one","two"], traces,
            extra_tokens=["start1","start2","A","X","B","Y"],
            num_registers=2, verbose=0)

    # @ut.skip("")
    def test_xsub(self):

        programs = {
        "one": """
        start1: mov r0 A
                sub sub2
                mov r0 C
                exit
        """,
        "two": """
                exit
        sub2:   mov r0 B
                ret
        """
        }
        trace = [
            {"r0": None, "r1": None}, # one start: mov
            {"r0": "A", "r1": None}, # one sub
            {"r0": "A", "r1": None}, # two mov
            {"r0": "B", "r1": None}, # two ret
            {"r0": "B", "r1": None}, # one mov
            {"r0": "C", "r1": None}, # one exit
        ]

        self._test(programs, ["one"], [trace],
            extra_tokens=["start1","sub2","A","B","C"],
            num_registers=2, verbose=0)

    # @ut.skip("")
    def test_memi(self):

        values = {"0": {"r0": "t0"}, "1": {"r0": "t1","r1":"t2"}}
        pointers = {"1": {"r0": "p0"}}
        memory = (pointers, values)

        program = """
        start:  drf r0
                rem r1
                mem r0
                prv
                rem r0
                mov r0 p0
                mov r1 t1
                drf r0
                rem r1
                exit
        """
        trace = [
            {"r0": "p0", "r1": None}, # start: drf r0
            {"r0": "p0", "r1": None}, # rem r1
            {"r0": "p0", "r1": "t2"}, # mem r0
            {"r0": "p0", "r1": "t2"}, # prv
            {"r0": "p0", "r1": "t2"}, # rem r0
            {"r0": "t0", "r1": "t2"}, # mov r0 p0
            {"r0": "p0", "r1": "t2"}, # mov r1 t1
            {"r0": "p0", "r1": "t1"}, # drf r0
            {"r0": "p0", "r1": "t1"}, # rem r1
            {"r0": "p0", "r1": "t2"}, # exit
            ]

        self._test({"test": program}, ["test"], [trace],
            memory=memory, extra_tokens=["start","t0","t1","t2","p0"],
            num_registers=2, verbose=0)

    # @ut.skip("")
    def test_dir_ptr(self):

        # Create pointer chain 0->1->2->0
        # Save memory [0:X, 1:Y, 2:Z]
        program = """
        start:  mov r1 P

                mov r0 X
                mem r0
                ref r1

                nxt
                mref r1
                mov r0 Y
                mem r0
                ref r1

                nxt
                mref r1
                mov r0 Z
                mem r0
                ref r1

                prv
                prv
                mref r1

                rem r0
                mdrf
                rem r0
                mdrf
                rem r0
                mdrf
                rem r0
                exit
        """
        trace = [
            {"r0": None, "r1": None}, # start: mov

            {"r0": None, "r1": "P"}, # mov
            {"r0": "X", "r1": "P"}, # mem
            {"r0": "X", "r1": "P"}, # ref

            {"r0": "X", "r1": "P"}, # nxt
            {"r0": "X", "r1": "P"}, # mref
            {"r0": "X", "r1": "P"}, # mov
            {"r0": "Y", "r1": "P"}, # mem
            {"r0": "Y", "r1": "P"}, # ref

            {"r0": "Y", "r1": "P"}, # nxt
            {"r0": "Y", "r1": "P"}, # mref
            {"r0": "Y", "r1": "P"}, # mov
            {"r0": "Z", "r1": "P"}, # mem
            {"r0": "Z", "r1": "P"}, # ref

            {"r0": "Z", "r1": "P"}, # prv
            {"r0": "Z", "r1": "P"}, # prv
            {"r0": "Z", "r1": "P"}, # mref

            {"r0": "Z", "r1": "P"}, # rem
            {"r0": "X", "r1": "P"}, # mdrf
            {"r0": "X", "r1": "P"}, # rem
            {"r0": "Y", "r1": "P"}, # mdrf
            {"r0": "Y", "r1": "P"}, # rem
            {"r0": "Z", "r1": "P"}, # mdrf
            {"r0": "Z", "r1": "P"}, # rem
            {"r0": "X", "r1": "P"}, # exit
            ]

        self._test({"test": program}, ["test"], [trace],
            extra_tokens=["start","S","P","X","Y","Z"],
            num_registers=2, verbose=0)

    # @ut.skip("")
    def test_many_regs(self):

        program = """
            start:  mov r0 5
                    exit
        """
        num_registers = 8
        reg_names = [("r%d" % d) for d in range(num_registers)]
        trace0 = {("r%d" % d): None for d in range(num_registers)}
        trace1 = dict(trace0)
        trace1["r0"] = "5"
        
        trace = [
            trace0, # start: mov r0 5
            trace1, # exit
            ]

        self._test({"test": program}, ["test"], [trace],
            num_registers=num_registers, verbose=0)

class RefVMTestCase(VMTestCase):
    def _make_vm(self, num_registers, programs, extra_tokens, verbose=False):
        return RefVM(["r%d"%r for r in range(num_registers)])

class NVMTestCase(VMTestCase):
    def _make_vm(self, num_registers, programs, extra_tokens, verbose=False):
        return make_scaled_nvm(
            register_names = ["r%d"%r for r in range(num_registers)],
            programs = programs,
            orthogonal=False,
            extra_tokens=extra_tokens,
            verbose=verbose)

class NVMOrthogonalTestCase(VMTestCase):
    def _make_vm(self, num_registers, programs, extra_tokens, verbose=False):
        return make_scaled_nvm(
            register_names = ["r%d"%r for r in range(num_registers)],
            programs = programs,
            orthogonal=True,
            extra_tokens=extra_tokens,
            verbose=verbose)

if __name__ == "__main__":
    test_suite = ut.TestLoader().loadTestsFromTestCase(RefVMTestCase)
    ut.TextTestRunner(verbosity=2).run(test_suite)

    test_suite = ut.TestLoader().loadTestsFromTestCase(NVMTestCase)
    ut.TextTestRunner(verbosity=2).run(test_suite)

    test_suite = ut.TestLoader().loadTestsFromTestCase(NVMOrthogonalTestCase)
    ut.TextTestRunner(verbosity=2).run(test_suite)
