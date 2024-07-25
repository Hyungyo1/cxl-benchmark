from numa_alloc import numa_alloc_tensor, numa_free_tensor, set_numa_interleave, unset_numa_interleave, check_tensor_node
import torch
import time
import argparse
import numpy as np
from queue import Queue
from threading import Thread
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
        # check_tensor_node(b, 6)
    
    # Generate data to be transferred
    if bwshare == 1:
        d_from = torch.rand(m, n).to(data_type).pin_memory()
        d_to = torch.zeros(m, n).to(data_type).to('cuda:0')
    elif bwshare == 2:
        d_from = realloc_to_numa(torch.rand(m, n).to(data_type))
        # check_tensor_node(d_from, 6)
        d_to = torch.zeros(m, n).to(data_type).to('cuda:0')
    
    def memcpy(queue):
        if bwshare == 1 or bwshare == 2:
            start = time.time()
            torch.cuda.synchronize()
            for j in range(math.ceil(0.005*m)):
                d_to.copy_(d_from, non_blocking=True)
                torch.cuda.synchronize()
            end = time.time()
            queue.put(end - start)
        else:
            queue.put(0)
    
    memcpy_queue = Queue()
    memcpy_thread = Thread(target=memcpy, args=(memcpy_queue,))
    memcpy_thread.start()

    # Measure time
    
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
            
    memcpy_thread.join()
    memcpy_time = memcpy_queue.get()

    # Accumulate duration
    compute_time = end - start
    if i > warmup - 1:
        durations_compute.append(compute_time)
        durations_memcpy.append(memcpy_time)

        print(f"Iteration {i - warmup}: output data type: {c.dtype}, Compute: {compute_time:.6f} s , Memcpy: {memcpy_time:.6f} s")
    else:
        print(f"Iteration {i - warmup}: output data type: {c.dtype}, Compute: {compute_time:.6f} s , Memcpy: {memcpy_time:.6f} s")

# Calculate average duration and GFLOPS
average_duration_compute = np.median(durations_compute)
average_duration_memcpy = np.median(durations_memcpy)
n_comp = 2 * bsz * m * n * k if bmm else 2 * m * n * k
gflops = n_comp / average_duration_compute / 10**9

print(f"Average Compute: {average_duration_compute:.6f} s")
print(f"Average Memcpy: {average_duration_memcpy:.6f} s")
print(f"Throughput: {gflops} (GFLOPS)")
