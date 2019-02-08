import sys
sys.path.append('../nvm')

from nvm import NVM
from activator import tanh_activator
from learning_rules import rehebbian
from tests import *

from syngen_nvm import *
from syngen import Network, Environment
from syngen import get_cpu, get_gpus, interrupt_engine

from datetime import datetime

class SyngenNVMTestCase(NVMTestCase):

    def _test(self, programs, names, traces, memory=None, tokens=[], num_registers=1, verbose=0):
        """
        Assemble all programs
        run all programs in list of names
        memory = (pointers, values)
        """

        vm = self._make_vm(num_registers, tokens=tokens)
        if memory is not None: vm.initialize_memory(*memory)
        
        vm.assemble(programs, verbose=0)

        ### BUILD NETWORK ####
        syn_net = SyngenNVM(vm.net)

        for name, trace in zip(names, traces):
            if verbose > 0:
                print()
                print(name)

            vm.load(name, trace[0])
            syn_net.initialize_activity(vm.net)

            ### SET UP VALIDATION CALLBACK ####
            class TestState:
                t = 0
                start = False

            def callback(layer_name, data):
                if layer_name == "gh":
                    if vm.net.layers["gh"].coder.decode(data) == "start":
                        TestState.start = True
                        TestState.t += 1
                    else:
                        TestState.start = False

                elif TestState.start:
                    state = vm.net.layers[layer_name].coder.decode(data)
                    if state != trace[TestState.t][layer_name]:
                        interrupt_engine()
                        self.assertTrue(False)

            ### BUILD ENVIRONMENG ####
            output_layers = vm.net.layers.keys() if verbose else []

            syn_env = SyngenEnvironment()
            syn_env.add_visualizer(output_layers)
            syn_env.add_printer(vm.net, output_layers)
            syn_env.add_checker(vm.net)
            syn_env.add_custom(
                ["gh"] + vm.register_names, "validator", callback)

            ### CHECK INITIAL STATE ###
            for r in vm.register_names:
                state = syn_net.decode_output(r, vm.net.layers[r].coder)
                self.assertTrue(state == trace[0][r])

            ### RUN SYNGEN ENGINE ###
            report = syn_net.run(syn_env, {
                "multithreaded" : False,
                "worker threads" : 0,
                "verbose" : verbose})
                
            if verbose:
                print(report)

        syn_net.free()


    def _make_vm(self, num_registers, tokens):
        orthogonal = False
        layer_shape = (16,16) if orthogonal else (32,32)
        pad = 0.0001
        activator, learning_rule = tanh_activator, rehebbian
        register_names = ["r%d"%r for r in range(num_registers)]

        return NVM(layer_shape,
            pad, activator, learning_rule, register_names,
            shapes={}, tokens=tokens, orthogonal=orthogonal)


if __name__ == "__main__":
    test_suite = ut.TestLoader().loadTestsFromTestCase(SyngenNVMTestCase)
    ut.TextTestRunner(verbosity=2).run(test_suite)
