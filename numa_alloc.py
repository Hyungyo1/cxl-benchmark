import ctypes
from ctypes import c_size_t, c_void_p, c_int, c_char
import torch
import numpy as np
import sys

# Load the shared library
lib = ctypes.CDLL('./libnuma_alloc.so')

# Define the function prototypes

numa_alloc_node = lib.numa_alloc_node
numa_alloc_node.argtypes = [c_size_t, c_int]
numa_alloc_node.restype = c_void_p

numa_alloc_interleave = lib.numa_alloc_interleave
numa_alloc_interleave.argtypes = [c_size_t]
numa_alloc_interleave.restype = c_void_p

numa_free_node = lib.numa_free_node
numa_free_node.argtypes = [c_void_p, c_size_t]
numa_free_node.restype = None

check_memory_node = lib.check_memory_node
check_memory_node.argtypes = [c_void_p, c_size_t]
check_memory_node.restype = None

lib.set_numa_interleave.argtypes = []
lib.set_numa_interleave.restype = ctypes.c_void_p

lib.unset_numa_interleave.argtypes = [ctypes.c_void_p]
lib.unset_numa_interleave.restype = None

def set_numa_interleave():
    return lib.set_numa_interleave()

def unset_numa_interleave(old_mask):
    lib.unset_numa_interleave(ctypes.c_void_p(old_mask))
    
def numa_alloc_tensor(shape, dtype):
    old_mask = set_numa_interleave()
    
    element_size = torch.tensor([], dtype=dtype).element_size()
    total_size = np.prod(shape) * element_size

    ptr = numa_alloc_interleave(total_size)
    if not ptr:
        print("Memory allocation failed")
        return None

    # Buffer creation from the raw pointer
    buffer = (c_char * total_size).from_address(ptr)
    
    # Create an untyped storage from the buffer
    storage = torch.UntypedStorage.from_buffer(buffer, dtype=dtype, byte_order=sys.byteorder)

    # Create a tensor from the untyped storage
    tensor = torch.tensor(storage, dtype=dtype).reshape(shape).pin_memory()
    
    unset_numa_interleave(old_mask)
    return tensor

def numa_free_tensor(tensor):
    ptr = ctypes.c_void_p(tensor.data_ptr())
    size = tensor.nelement() * tensor.element_size()
    numa_free_node(ptr, c_size_t(size))

def check_tensor_node(tensor, num_pages):
    ptr = ctypes.c_void_p(tensor.data_ptr())
    check_memory_node(ptr, num_pages)