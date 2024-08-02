from numa_alloc import numa_alloc_tensor, numa_free_tensor, set_numa_interleave, unset_numa_interleave, check_tensor_node
import torch
import time
import argparse
import numpy as np
import math
import os
import psutil
import torch.cuda as cuda

from multiprocessing import Process, Queue, Barrier
# from threading import Thread, Barrier
# from queue import Queue

def realloc_to_numa(tensor):
    numa_tensor = numa_alloc_tensor(tensor.shape, tensor.dtype)
    if numa_tensor is not None:
        numa_tensor.copy_(tensor)
        del tensor
        return numa_tensor
    else:
        raise MemoryError("Fail to allocate CXL memory!")

# Argument parsing
parser = argparse.ArgumentParser(description="Matrix multiplication with optional settings.")
parser.add_argument('--bmm', action='store_true', help="Use batched matrix multiplication.")
parser.add_argument('--amx', action='store_true', help="Use AMX (Advanced Matrix Extensions).")
parser.add_argument('--bsz', type=int, default=5040, help="Batch size (only used if --bmm is set).")
parser.add_argument('--m', type=int, default=2048, help="Number of rows of the first matrix.")
parser.add_argument('--n', type=int, default=2048, help="Number of columns of the first matrix and rows of the second matrix.")
parser.add_argument('--k', type=int, default=2048, help="Number of columns of the second matrix.")
parser.add_argument('--iter', type=int, default=5, help="Number of iterations to repeat the process.")
parser.add_argument('--warmup', type=int, default=2, help="Number of iterations to repeat the process.")
parser.add_argument('--cxl', action='store_true', help="Use CXL to store Param/KV cache.")
parser.add_argument('--bwshare', type=int, default=0, help="0-not enabled, 1-copy from DDR to GPU, 2-copy from CXL to GPU")
args = parser.parse_args()

# Set variables based on arguments
bmm = args.bmm
amx = args.amx
bsz = args.bsz if bmm else 1
m = args.m
n = args.n
k = args.k
iterations = args.iter
warmup = args.warmup
cxl = args.cxl
bwshare = args.bwshare

data_type = torch.bfloat16 if amx else torch.float32

def memcpy_process(barrier, queue, data_type, args, iterations):
    p = psutil.Process(os.getpid())
    try:
        if args.bwshare == 2:
            p.cpu_affinity([40])
        else:
            p.cpu_affinity([0])
    except Exception as e:
        print(f"Failed to set affinity: {str(e)}")
    parent_priority = psutil.Process(os.getppid()).nice()
    p.nice(parent_priority)
    priority = psutil.Process(os.getppid()).nice()
    affinity = p.cpu_affinity()
    print(f"[Transfer Process] Prio: {priority}, Affi: {affinity}")

    transfer_stream = cuda.Stream()
    if args.bwshare == 1:
        d_from = torch.rand(args.bsz * args.m * args.n).to(data_type).pin_memory()
    if args.bwshare == 2:
        d_from = realloc_to_numa(torch.rand(args.bsz * args.m * args.n).to(data_type))
    
    barrier.wait()
    if args.bwshare == 1 or args.bwshare == 2:
        print("Tensor D:")
        check_tensor_node(d_from, 4)

    for i in range(iterations):
        barrier.wait()
        start = time.time()
        if args.bwshare == 1 or args.bwshare == 2:
            with cuda.stream(transfer_stream):
                for j in range(80):
                    d_to = d_from.to('cuda', non_blocking=True)
                    torch.cuda.synchronize()
        end = time.time()
        queue.put(end - start)
        barrier.wait()

# Repeat the time measurement process
durations_compute = []
durations_memcpy = []

p = psutil.Process(os.getpid())
p.nice(0)
priority = psutil.Process(os.getppid()).nice()
affinity = p.cpu_affinity()
print(f"[Compute  Process] Prio: {priority}, Affi: {affinity}")

barrier = Barrier(2)
memcpy_queue = Queue()
memcpy_proc = Process(target=memcpy_process, args=(barrier, memcpy_queue, data_type, args, iterations))
memcpy_proc.start()

if bmm:
    a = torch.rand(args.bsz, args.m, args.n).to(data_type)
    b = torch.rand(args.bsz, args.n, args.k).to(data_type)
else:
    a = torch.rand(args.m, args.n).to(data_type)
    b = torch.rand(args.n, args.k).to(data_type)
if cxl:
    b = realloc_to_numa(b)

print("Tensor A:")
check_tensor_node(a, 4)
print("Tensor B:")
check_tensor_node(b, 4)
barrier.wait()

for i in range(iterations):

    barrier.wait()

    if args.amx:
        with torch.cpu.amp.autocast():
            if args.bmm:
                start = time.time()
                c = torch.bmm(a, b)
                end = time.time()
            else:
                start = time.time()
                c = torch.matmul(a, b)
                end = time.time()
    else:
        if args.bmm:
            start = time.time()
            c = torch.bmm(a, b)
            end = time.time()
        else:
            start = time.time()
            c = torch.matmul(a, b)
            end = time.time()

    barrier.wait()

    compute_time = end - start
    memcpy_time = memcpy_queue.get()

    if i > warmup - 1:
        durations_compute.append(compute_time)
        durations_memcpy.append(memcpy_time)
        print(f"Iteration {i - warmup}: output data type: {data_type}, Compute: {compute_time:.6f} s , Memcpy: {memcpy_time:.6f} s")
    # else:
    #     print(f"Iteration {i - warmup}: output data type: {data_type}, Compute: {compute_time:.6f} s , Memcpy: {memcpy_time:.6f} s")

memcpy_proc.join()

# Calculate average duration and GFLOPS
average_duration_compute = np.median(durations_compute)
average_duration_memcpy = np.median(durations_memcpy)
n_comp = 2 * bsz * m * n * k if bmm else 2 * m * n * k
gflops = n_comp / average_duration_compute / 10**9

print(f"Average Compute: {average_duration_compute:.6f} s")
print(f"Average Memcpy: {average_duration_memcpy:.6f} s")
print(f"Throughput: {gflops:.6f} (GFLOPS)")
