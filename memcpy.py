import multiprocessing
import threading
import numpy as np
import torch
import os
import time
import psutil

def to_gpu(queue, barrier, size, repeat):
    p = psutil.Process(os.getpid())
    parent_priority = psutil.Process(os.getppid()).nice()
    p.nice(parent_priority)

    data = torch.randn(size, dtype=torch.float32, device='cpu', pin_memory=True)
    
    barrier.wait()

    for i in range(repeat):
        start = time.time()
        gpu_data = data.to('cuda', non_blocking=True)
        torch.cuda.synchronize()
        end = time.time()
        queue.put(end - start)
        barrier.wait()

def to_cpu(queue, barrier, size, repeat):
    p = psutil.Process(os.getpid())
    parent_priority = psutil.Process(os.getppid()).nice()
    p.nice(parent_priority)

    data = np.random.rand(size).astype(np.float32)
    num_threads = os.cpu_count()

    def read_data(data, i):
        stride = size // num_threads
        start = i * stride
        end = (i + 1) * stride if i != num_threads - 1 else size
        result = np.sum(data[start:end])
    
    barrier.wait()
    
    for i in range(repeat):
        threads = []
        start = time.time()
        for j in range(num_threads):
            thread = threading.Thread(target=read_data, args=(data, j,))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        end = time.time()
        queue.put(end - start)
        barrier.wait()

if __name__ == '__main__':
    size = 1024 * 1024 * 512
    repeat = 10
    warmup = 5
    barrier = multiprocessing.Barrier(2)
    cpu_queue = multiprocessing.Queue()
    gpu_queue = multiprocessing.Queue()

    # Creating the multiprocessing Process objects
    process_cpu = multiprocessing.Process(target=to_cpu, args=(cpu_queue, barrier, size, repeat,))
    process_gpu = multiprocessing.Process(target=to_gpu, args=(gpu_queue, barrier, size, repeat,))

    # Starting the processes
    process_gpu.start()
    process_cpu.start()

    cpu_bw_l = []
    gpu_bw_l = []
    for i in range(repeat):
        cpu_time = cpu_queue.get()
        gpu_time = gpu_queue.get()
        cpu_bw = size / (1024 * 1024 * 1024) / 4 / cpu_time
        gpu_bw = size / (1024 * 1024 * 1024) / 4 / gpu_time
        print(f"[Iternation {i-warmup}] CPU Bandwidth: {cpu_bw:.3f} GB/s; GPU Bandwidth: {gpu_bw:.3f} GB/s")
        if i >= warmup:
            cpu_bw_l.append(cpu_bw)
            gpu_bw_l.append(gpu_bw)

    # Wait for both processes to complete
    process_gpu.join()
    process_cpu.join()

    cpu_bw_ave = np.median(cpu_bw_l)
    gpu_bw_ave = np.median(gpu_bw_l)

    print(f"[Average] CPU Bandwidth: {cpu_bw_ave:.3f} GB/s; GPU Bandwidth: {gpu_bw_ave:.3f} GB/s")