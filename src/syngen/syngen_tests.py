import sys
sys.path.append('../nvm')

from nvm import NVM
from activator import tanh_activator
from learning_rules import rehebbian
from tests import *

from syngen_nvm import make_syngen_network, make_syngen_environment, init_syngen_nvm
from syngen import Network, Environment
from syngen import get_cpu, get_gpus
from syngen import set_suppress_output, set_warnings, set_debug

class SyngenNVMTestCase(NVMTestCase):

    def _test(self, programs, names, traces, memory=None, tokens=[], num_registers=1, verbose=0):
        """
        Assemble all programs
        run all programs in list of names
        memory = (pointers, values)
        """

        set_suppress_output(False)
        set_warnings(False)
        set_debug(False)

        vm = self._make_vm(num_registers, tokens=tokens)
        if memory is not None: vm.initialize_memory(*memory)
        
        vm.assemble(programs, verbose=0)

        for name, trace in zip(names, traces):
            vm.load(name, trace[0])

            structure, connections = make_syngen_network(vm.net)
            layers = vm.net.layers.keys()
            modules = make_syngen_environment(vm.net, run_nvm=True,
                viz_layers=layers, print_layers=layers, stat_layers=[], read=True)

            net = Network({"structures" : [structure], "connections" : connections})
            env = Environment({"modules" : modules})

            init_syngen_nvm(vm.net, net)

            print(net.run(env, {"multithreaded" : False,
                                    "worker threads" : 0,
                                    "device" : get_gpus()[0],
                                    "iterations" : 10000,
                                    "refresh rate" : 0,
                                    "verbose" : True,
                                    "learning flag" : False}))
    
            # Delete the objects
            del net
            del env
            continue
    
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

    def _make_vm(self, num_registers, tokens):
        orthogonal = True
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
