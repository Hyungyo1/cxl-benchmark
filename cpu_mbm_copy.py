from numa_alloc import numa_alloc_tensor, numa_free_tensor, set_numa_interleave, unset_numa_interleave, check_tensor_node
import torch
import time
import argparse
import numpy as np
from queue import Queue
from threading import Thread
# import multiprocessing 
# multiprocessing.set_start_method('spawn', force=True)
from multiprocessing import Process
import math

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
# Add for 7.c
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

# Repeat the time measurement process
durations_compute = []
durations_memcpy = []
for i in range(iterations):
    # Generate random tensors
    if bmm:
        a = torch.rand(bsz, m, n).to(data_type)
        b = torch.rand(bsz, n, k).to(data_type)
    else:
        a = torch.rand(m, n).to(data_type)
        b = torch.rand(n, k).to(data_type)
    
    if cxl:
        b = realloc_to_numa(b)

    # Generate data to be transferred
    if bwshare == 1:
        d_from = torch.rand(bsz, m, n).to(data_type).pin_memory()
        d_to = torch.zeros(bsz, m, n).to(data_type).to('cuda:0')
    elif bwshare == 2:
        d_from = realloc_to_numa(torch.rand(bsz, m, n).to(data_type))
        d_to = torch.zeros(bsz, m, n).to(data_type).to('cuda:0')

    def compute(queue):
        if amx:
            with torch.cpu.amp.autocast():
                if bmm:
                    start = time.time()
                    c = torch.bmm(a, b)
                    end = time.time()
                else:
                    start = time.time()
                    c = torch.matmul(a, b)
                    end = time.time()
        else:
            if bmm:
                start = time.time()
                c = torch.bmm(a, b)
                end = time.time()
            else:
                start = time.time()
                c = torch.matmul(a, b)
                end = time.time()
        print("Test 3.2")
        queue.put(end - start)
        queue.put(c.dtype)

    def memcpy(queue):
        start = time.time()
        end = time.time()
        queue.put(end - start)
    
    print("Test 1")
    memcpy_queue = Queue()
    compute_queue = Queue()
    memcpy_thread = Process(target=memcpy, args=(compute_queue,))
    compute_thread = Process(target=compute, args=(compute_queue,))
    print("Test 2")
    memcpy_thread.start()
    compute_thread.start()
    print("Test 3")
    memcpy_thread.join()
    compute_thread.join()
    print("Test 4")

    memcpy_time = memcpy_queue.get()
    compute_time = compute_queue.get()
    out_dtype = compute_queue.get()

    if i > warmup - 1:
        durations_memcpy.append(memcpy_time)
        durations_compute.append(compute_time)

        print(f"Iteration {i - warmup}: output data type: {out_dtype}, Compute Duration: {compute_time:.6f} seconds, Memcpy Duration {memcpy_time:.6f} seconds")
        # print(f"Iteration {i - warmup}: output data type: {out_dtype}, Compute Duration: {compute_time:.6f} seconds")

# Calculate average duration and GFLOPS
average_duration_compute = np.median(durations_compute)
average_duration_memcpy = np.median(durations_memcpy)
n_comp = 2 * bsz * m * n * k if bmm else 2 * m * n * k
gflops = n_comp / average_duration_compute / 10**9

print(f"Average Compute Duration: {average_duration_compute:.6f} seconds")
print(f"Average Memcpy Duration: {average_duration_memcpy:.6f} seconds")
print(f"Throughput: {gflops} (GFLOPS)")
