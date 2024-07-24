from numa_alloc import numa_alloc_tensor, numa_free_tensor, set_numa_interleave, unset_numa_interleave, check_tensor_node
import torch
import time
import argparse
import numpy as np

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

data_type = torch.bfloat16 if amx else torch.float32

# Repeat the time measurement process
duration_list = []
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

    # Accumulate duration
    duration = end - start
    if i > warmup - 1:
        duration_list.append(duration)

        print(f"Iteration {i - warmup}: output data type: {c.dtype}, Duration: {duration:.6f} seconds")

# Calculate average duration and GFLOPS
average_duration = np.median(duration_list)
n_comp = 2 * bsz * m * n * k if bmm else 2 * m * n * k
gflops = n_comp / average_duration / 10**9

print(f"Average Duration: {average_duration:.6f} seconds")
print(f"Throughput: {gflops} (GFLOPS)")
