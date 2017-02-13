import multiprocessing as mp

def run_sub(pipe_to_sup):
    print(pipe_to_sup.recv())
    print(pipe_to_sup.recv())

if __name__ == "__main__":
    pipe_to_sub, pipe_for_sub = mp.Pipe()
    sub_process = mp.Process(target=run_sub, args=(pipe_for_sub,))
    pipe_to_sub.send("hey1")
    pipe_to_sub.send("hey2")
    sub_process.start()
    sub_process.join()
    
