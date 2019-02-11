import sys
sys.path.append('../nvm')

from nvm import NVM
from activator import tanh_activator
from learning_rules import rehebbian
from tests import *

from syngen_nvm import *
from op_net import *
from syngen import Network, Environment
from syngen import get_cpu, get_gpus, interrupt_engine

from datetime import datetime
import numpy as np

def test_syngen(tester, programs, names, traces, memory=None, tokens=[], num_registers=1, verbose=0):
    """
    Assemble all programs
    run all programs in list of names
    memory = (pointers, values)
    """

    vm = tester._make_vm(num_registers, tokens=tokens)
    if memory is not None: vm.initialize_memory(*memory)

    vm.assemble(programs, verbose=0)

    ### BUILD NETWORK ####
    syn_net = tester._make_syngen_nvm(vm)

    for name, trace in zip(names, traces):
        if verbose > 0:
            print()
            print(name)

        vm.load(name, trace[0])
        syn_net.initialize_activity(vm.net)

        ### SET UP VALIDATION CALLBACK ####
        class TestState:
            t = 0
            unk_count = 0
            start = False
            failed = False

        def callback(layer_name, data):
            if layer_name == "gh":
                state = vm.net.layers["gh"].coder.decode(data)

                if state == "start":
                    TestState.start = True
                    TestState.t += 1
                    #print("\n%d" % TestState.t)
                else:
                    TestState.start = False

                if state == "?":
                    if TestState.unk_count > 20:
                        print("Gate mechanism derailed!")
                        interrupt_engine()
                        TestState.failed = True
                    TestState.unk_count += 1
                else:
                    TestState.unk_count = 0

            elif TestState.start:
                state = vm.net.layers[layer_name].coder.decode(data)
                trace_t = trace[TestState.t]

                #print(layer_name, state)

                if layer_name in trace_t and state != trace_t[layer_name]:
                    print("Trace mismatch!")
                    print(TestState.t, layer_name, state, trace_t[layer_name])
                    interrupt_engine()
                    TestState.failed = True

        ### BUILD ENVIRONMENG ####
        output_layers = vm.net.layers.keys() if verbose else []

        syn_env = tester._make_syngen_env(vm)
        syn_env.add_visualizer("nvm", output_layers)
        syn_env.add_printer(vm.net, output_layers)
        syn_env.add_custom("nvm",
            ["gh"] + vm.register_names, "validator", callback)

        ### CHECK INITIAL STATE ###
        for r in vm.register_names:
            if r in trace[0]:
                state = syn_net.decode_output(r, vm.net.layers[r].coder)
                if state != trace[0][r]:
                    syn_net.free()
                    print("Trace mismatch!")
                    print(0, r, state, trace[0][r])
                    return False

        ### RUN SYNGEN ENGINE ###
        report = syn_net.run(syn_env, {
            "multithreaded" : False,
            "worker threads" : 0,
            "verbose" : verbose})

        if verbose:
            print(report)

        if TestState.failed:
            syn_net.free()
            return False

    syn_net.free()
    return True

class SyngenNVMTestCase(NVMTestCase):

    def _make_syngen_nvm(self, vm):
        return SyngenNVM(vm.net)

    def _make_syngen_env(self, vm):
        syn_env = SyngenEnvironment()
        syn_env.add_checker(vm.net)
        return syn_env

    def _test(self, programs, names, traces,
            memory=None, tokens=[], num_registers=1, verbose=0):
        self.assertTrue(test_syngen(
            self, programs, names, traces,
            memory, tokens, num_registers, verbose))

    def _make_vm(self, num_registers, tokens):
        orthogonal = False
        layer_shape = (16,16) if orthogonal else (32,32)
        pad = 0.0001
        activator, learning_rule = tanh_activator, rehebbian
        register_names = ["r%d"%r for r in range(num_registers)]

        return NVM(layer_shape,
            pad, activator, learning_rule, register_names,
            shapes={}, tokens=tokens, orthogonal=orthogonal)

class SyngenNVMArithTestCase(ut.TestCase):
    opdef = make_arith_opdef(in_range=range(-1,10), out_range=range(-9,10))

    def _make_syngen_nvm(self, vm):
        return SyngenNVM(vm.net,
            [OpNet(vm.net, self.opdef, ["r0", "r1"], ["r0", "r1"], "r2")])

    def _make_syngen_env(self, vm):
        return SyngenEnvironment()

    def _test(self, programs, names, traces,
            memory=None, tokens=[], num_registers=1, verbose=0):
        self.assertTrue(test_syngen(
            self, programs, names, traces,
            memory, tokens, num_registers, verbose))

    def _make_vm(self, num_registers, tokens):
        orthogonal = True
        layer_shape = (16,16) if orthogonal else (32,32)
        pad = 0.0001
        activator, learning_rule = tanh_activator, rehebbian
        register_names = ["r%d"%r for r in range(num_registers)]

        return NVM(layer_shape,
            pad, activator, learning_rule, register_names,
            shapes={"gh" : (32,32)}, tokens=tokens, orthogonal=orthogonal)

    # @ut.skip("")
    def test_add_simple(self):

        program = """
        start:  mov r0 1
                mov r1 2
                mov r2 +
                nop
                mov r2 A
                nop
                mov r2 +
                nop
                mov r0 9
                mov r2 +
                nop
                mov r0 -1
                mov r2 +
                nop
                mov r0 A
                mov r2 +
                nop
                exit
        """
        trace = [
            {"r0": None   , "r1": None   ,"r2": None   },
            {"r0": "1"    , "r1": None   ,"r2": None   },
            {"r0": "1"    , "r1": "2"    ,"r2": None   },
            {"r0": "1"    , "r1": "2"    ,"r2": "+"    },
            {"r0": "0"    , "r1": "3"    ,"r2": "null" }, # 1 + 2 = 3
            {"r0": "0"    , "r1": "3"    ,"r2": "A"    },
            {"r0": "0"    , "r1": "3"    ,"r2": "A"    }, # no change
            {"r0": "0"    , "r1": "3"    ,"r2": "+"    },
            {"r0": "0"    , "r1": "3"    ,"r2": "null" }, # 0 + 3 = 3
            {"r0": "9"    , "r1": "3"    ,"r2": "null" },
            {"r0": "9"    , "r1": "3"    ,"r2": "+"    },
            {"r0": "1"    , "r1": "2"    ,"r2": "null" }, # 9 + 3 = 12
            {"r0": "-1"   , "r1": "2"    ,"r2": "null" },
            {"r0": "-1"   , "r1": "2"    ,"r2": "+"    },
            {"r0": "0"    , "r1": "1"    ,"r2": "null" }, # -1 + 2 = 1
            {"r0": "A"    , "r1": "1"    ,"r2": "null" },
            {"r0": "A"    , "r1": "1"    ,"r2": "+"    },
            {"r0": "null" , "r1": "null" ,"r2": "null" }, # A + 1 = ?
        ]

        self._test({"test":program}, ["test"], [trace],
            tokens = ["A"] + self.opdef.tokens,
            num_registers=3, verbose=0)


    def _test_bin_op(self, opcode):
        f0,f1 = self.opdef.operations[opcode]

        values = {str(i): {"r0": x,"r1": x}
            for i,x in enumerate(self.opdef.in_ops)}
        values[str(len(self.opdef.in_ops))] = {"r0": "null","r1": "null"}
        pointers = {"0": {"r3": "arr"}}
        memory = (pointers, values)

        # Advances pointers along memory, performing the
        #   operation on all pairs of digits from 0-9.
        #
        # for x in range(10):
        #     for y in range(10):
        #         x + y
        program = """
        start:  mov r3 arr
                drf r3
                mov r3 ptr
                ref r3
                mov r4 ptr
                ref r4

        loop:   drf r3
                rem r1
                nxt
                ref r3
                cmp r1 null
                jie break

                drf r4
                rem r0
                mov r2 %s
                nop

                jmp loop

        break:  drf r4
                nxt
                ref r4
                rem r0
                cmp r0 null
                jie end

                mov r3 arr
                drf r3
                mov r3 ptr
                ref r3
                jmp loop

        end:    exit
        """ % opcode
        trace = [
            {},
            {"r3": "arr" },
            {"r3": "arr" },
            {"r3": "ptr" },
            {"r3": "ptr" },
            {"r3": "ptr", "r4": "ptr" },
            {"r3": "ptr", "r4": "ptr" },
        ]

        for i,x in enumerate(self.opdef.in_ops):
            for y in self.opdef.in_ops:
                try: v0,v1 = f0(x,y), f1(x,y)
                except: v0,v1 = 'null','null'

                # Loop iteration without break
                trace += [
                    {},                                    # drf
                    {"r1": y },                            # rem
                    {},                                    # nxt
                    {},                                    # ref
                    {"co": "false"},                       # cmp
                    {},                                    # jie

                    {},                                    # drf
                    {"r0": x , "r1": y },                  # rem
                    {"r0": x , "r1": y ,"r2": opcode },    # mov
                    {"r0": v0 , "r1": v1 ,"r2": "null" },  # nop

                    {},                                    # jmp
                ]

            # Last loop iteration with break, post-loop code
            next_op = (
                "null" if i == len(self.opdef.in_ops)-1
                       else self.opdef.in_ops[i+1])
            trace += [
                {},                                        # drf
                {"r1": "null" },                           # rem
                {},                                        # nxt
                {},                                        # ref
                {"co": "true"},                            # cmp
                {},                                        # jie

                {},                                        # drf
                {},                                        # nxt
                {},                                        # ref
                {"r0": next_op},                           # rem
                {"cmp": "true"},                           # cmp
                {},                                        # jie

                {"r3": "arr"},                             # mov
                {},                                        # drf
                {"r3": "ptr"},                             # mov
                {},                                        # ref
                {},                                        # jmp
            ]
        trace.append({}) # exit

        self._test({"test":program}, ["test"], [trace], memory = memory,
            tokens = ["arr", "ptr"] + self.opdef.tokens,
            num_registers=5, verbose=0)


    # @ut.skip("")
    def test_add(self):
        self._test_bin_op("+")

    # @ut.skip("")
    def test_sub(self):
        self._test_bin_op("-")

    # @ut.skip("")
    def test_mult(self):
        self._test_bin_op("*")

    # @ut.skip("")
    def test_div(self):
        self._test_bin_op("/")



if __name__ == "__main__":
    test_suite = ut.TestLoader().loadTestsFromTestCase(SyngenNVMTestCase)
    ut.TextTestRunner(verbosity=2).run(test_suite)

    test_suite = ut.TestLoader().loadTestsFromTestCase(SyngenNVMArithTestCase)
    ut.TextTestRunner(verbosity=2).run(test_suite)
