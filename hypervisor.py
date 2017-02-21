"""
HV (main, interactive commands)
<-> IO, source code, nvm directives
NVM (network, encoding, build tools)
<-> network state, viz directives
VIZ (viz)
"""
import sys
import multiprocessing as mp
import nvm

class NotRunningError(Exception):
    pass

class Hypervisor:
    def __init__(self):
        self.startup()
    def startup(self):
        self.nvm_pipe, other_end = mp.Pipe()
        self.nvm_process = mp.Process(target=run_nvm, args=(other_end,))
        self.nvm_process.start()
        self.running = True
    def exchange(self, message):
        if not self.running: raise NotRunningError()
        self.nvm_pipe.send(message)
        response = self.nvm_pipe.recv()
        return response
    def print_nvm(self):
        response = self.exchange('print')
        print(response)
    def show(self):
        response = self.exchange('show')
        print(response)
    def hide(self):
        response = self.exchange('hide')
        print(response)
    def shutdown(self):
        self.exchange('shutdown')
        self.nvm_process.join()
        self.nvm_pipe = None
        self.nvm_process = None
        self.running = False
    def quit(self):
        if self.running: self.shutdown()
        sys.exit(0)

def run_nvm(hv_pipe):
    # Init NVM
    vm = nvm.mock_nvm()
    done = False
    while not done:
        # step the network
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
            if message == 'shutdown':
                vm.hide() # shutdown visualizer if running
                done = True
                hv_pipe.send('shutdown')

if __name__ == '__main__':

    # Start hypervisor
    hv = Hypervisor()
    hv.print_nvm()
    hv.show()
    # hv.terminate()

# import multiprocessing as mp
# import sys
# import time
# import refvm

# def _run_vm(vm, io, period, pipe_to_hv):
#     while True:
#         start_time = time.time()
#         # flush pipes
#         if pipe_to_hv.poll():
#             message = pipe_to_hv.recv()
#             if message == 'q':
#                 vm.hide_gui()
#                 break
#             if message == 'd':
#                 print('%s seconds elapsed'%(time.time()-start_time))
#                 vm.disp()
#                 pipe_to_hv.send('done')
#             if message == 'p':
#                 data = pipe_to_hv.recv()
#                 io.put(data)
#                 pipe_to_hv.send('done')
#             if message == 'k':
#                 data = io.peek()
#                 pipe_to_hv.send(data)
#             if message == 's':
#                 vm.show_gui()
#                 pipe_to_hv.send('done')
#             if message == 'h':
#                 vm.hide_gui()
#                 pipe_to_hv.send('done')
#         # step the vm
#         vm.tick()
#         tick_time = time.time()-start_time
#         if period > tick_time:
#             time.sleep(period - tick_time)

# class Hypervisor:
#     def __init__(self):
#         pass
#     def start(self, vm, io, period=1):
#         self.pipe_to_vm, pipe_to_hv = mp.Pipe()
#         self.vm_process = mp.Process(target=_run_vm, args=(vm, io, period, pipe_to_hv))
#         self.vm_process.start()
#     def stop(self):
#         print('Stopping...')
#         self.pipe_to_vm.send('q')
#         self.vm_process.join()
#         print('Stopped.')
#         exit()
#     def disp(self):
#         print('State:')
#         self.pipe_to_vm.send('d')
#         self.pipe_to_vm.recv() # confirm finished
#     def show(self):
#         self.pipe_to_vm.send('s')
#         self.pipe_to_vm.recv() # confirm finished
#     def hide(self):
#         self.pipe_to_vm.send('h')
#         self.pipe_to_vm.recv() # confirm finished
#     def put(self, data):
#         print('Putting %s'%data)
#         self.pipe_to_vm.send('p')
#         self.pipe_to_vm.send(data)
#         self.pipe_to_vm.recv() # confirm finished
#     def peek(self):
#         self.pipe_to_vm.send('k')
#         print(self.pipe_to_vm.recv())

# listener_program = """
# set NIL {0} # NIL for exiting loop
# loop: get rvmio {1} # get input
# compare {0} {1} {2} # compare with NIL
# nor {2} {2} {3} # true if not NIL
# jump {3} loop # if not NIL, repeat
# put rvmio {0}
# # end
# nop
# """

# echo_program = """
# set NIL {0} # NIL for exiting loop
# loop: get rvmio {1} # get input
# put rvmio {1} # echo
# compare {0} {1} {2} # compare with NIL
# nor {2} {2} {3} # true if not NIL
# jump {3} loop # if not NIL, repeat
# put rvmio {0}
# # end
# nop
# """

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
