"""
HV (main, interactive commands)
<-> IO, source code, nvm directives
NVM (network, encoding, build tools)
<-> network state, viz directives
VIZ (viz)
"""
import sys, time
import multiprocessing as mp
import nvm

class NotRunningError(Exception):
    pass

class Hypervisor:
    def startup(self, period=1):
        self.nvm_pipe, other_end = mp.Pipe()
        self.nvm_process = mp.Process(target=run_nvm, args=(other_end, period))
        self.nvm_process.start()
        self.running = True
    def exchange_with_nvm(self, message):
        if not self.running: raise NotRunningError()
        self.nvm_pipe.send(message)
        response = self.nvm_pipe.recv()
        return response
    def print_nvm(self):
        response = self.exchange_with_nvm('print')
        print(response)
    def show(self):
        response = self.exchange_with_nvm('show')
        print(response)
    def hide(self):
        response = self.exchange_with_nvm('hide')
        print(response)
    def input(self, token):
        ready = self.exchange_with_nvm('input')
        response = self.exchange_with_nvm(token)
        print(ready,response)
    def output(self):
        response = self.exchange_with_nvm('output')
        return response
    def set_operation(self, opcode, *operands):
        ready = self.exchange_with_nvm('set_operation')
        opcode_response = self.exchange_with_nvm(opcode)
        for operand in operands:
            operand_response = self.exchange_with_nvm(operand)
        response = self.exchange_with_nvm('end operation')
    def shutdown(self):
        self.exchange_with_nvm('shutdown')
        self.nvm_process.join()
        self.nvm_pipe = None
        self.nvm_process = None
        self.running = False
    def quit(self):
        if self.running: self.shutdown()
        sys.exit(0)

def run_nvm(hv_pipe, period):
    # Init NVM
    vm = nvm.mock_nvm(layer_size=24)
    done = False
    while not done:
        # step the network
        start_time = time.time()
        vm.tick()
        # process next message
        if hv_pipe.poll():
            message = hv_pipe.recv()
            if message == 'print':
                string = 'happy'
                hv_pipe.send(string)
            if message == 'show':
                vm.show()
                hv_pipe.send('showing')
            if message == 'hide':
                vm.hide()
                hv_pipe.send('hiding')
            if message == 'input':
                hv_pipe.send('accepting input')
                token = hv_pipe.recv()
                vm.set_input(token, 'stdio', from_human_readable=True)
                hv_pipe.send('received %s'%token)
            if message == 'output':
                token = vm.get_output('stdio',to_human_readable=True)
                hv_pipe.send(token)
            if message == 'set_operation':
                hv_pipe.send('accepting operation')
                opcode = hv_pipe.recv()
                hv_pipe.send('recieved opcode')
                operands = []
                while True:
                    message = hv_pipe.recv()
                    hv_pipe.send('received operand')
                    if message == 'end operation': break
                    operands.append(message)
                vm.set_operation(opcode, *operands)
                hv_pipe.send('received operation')
            if message == 'shutdown':
                vm.hide() # shutdown visualizer if running
                done = True
                hv_pipe.send('shutdown')
        # wait up to period
        duration = time.time() - start_time
        if duration < period:
            time.sleep(period - duration)

if __name__ == '__main__':

    period = .1
    
    # Start hypervisor
    hv = Hypervisor()
    hv.startup(period=period)
    hv.show()
    time.sleep(period)
    hv.input('TRUE')
    time.sleep(period)
    hv.set_operation('set','NIL','{0}')
    time.sleep(period)
    # hv.shutdown()

listener_program = """
set NIL {0} # NIL for exiting loop
loop: get rvmio {1} # get input
compare {0} {1} {2} # compare with NIL
nor {2} {2} {3} # true if not NIL
jump {3} loop # if not NIL, repeat
put rvmio {0}
# end
nop
"""

echo_program = """
set NIL {0} # NIL for exiting loop
loop: get rvmio {1} # get input
put rvmio {1} # echo
compare {0} {1} {2} # compare with NIL
nor {2} {2} {3} # true if not NIL
jump {3} loop # if not NIL, repeat
put rvmio {0}
# end
nop
"""

# if __name__ == '__main__':
#     rvm = refvm.RefVM()
#     rvmio = refvm.RefIODevice(rvm.machine_readable, rvm.human_readable)
#     rvm.install_device('rvmio',rvmio)

#     # assembly_code = listener_program
#     assembly_code = echo_program
#     object_code, label_table = rvm.assemble(assembly_code)
#     rvm.load(object_code, label_table)

#     hv = Hypervisor()
#     print('Starting...')
#     hv.start(rvm, rvmio, period=1.0/10)
#     hv.show()
