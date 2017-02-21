import numpy as np
import multiprocessing as mp

def run_sub(pipe_to_sup):
    print(pipe_to_sup.recv())
    print(pipe_to_sup.recv())
    message = pipe_to_sup.recv_bytes()
    print(message)
    print(message == "test")
    print(np.fromstring(message,dtype=np.uint8))

if __name__ == "__main__":
    pipe_to_sub, pipe_for_sub = mp.Pipe()
    sub_process = mp.Process(target=run_sub, args=(pipe_for_sub,))
    pipe_to_sub.send("hey1")
    pipe_to_sub.send("hey2")
    arr = np.array([3,2,1],dtype=np.uint8)
    # pipe_to_sub.send_bytes(arr.tobytes())
    # pipe_to_sub.send("test recv_bytes() against send()")
    pipe_to_sub.send("test")
    sub_process.start()
    sub_process.join()
    
    test = {}
    test[arr.tobytes()] = 3
    print(test[arr.tobytes()])
    print(test.keys())
