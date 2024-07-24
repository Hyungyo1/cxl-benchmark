# GEMV

echo "GEMV DDR"

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --bmm --bsz 3072 --m 1 --n 128 --k 64 --iter 20 --warmup 10
OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --bmm --bsz 3072 --m 1 --n 128 --k 256 --iter 20 --warmup 10
OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --bmm --bsz 3072 --m 1 --n 128 --k 1024 --iter 20 --warmup 10

echo "GEMV CXL"

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --bmm --bsz 3072 --m 1 --n 128 --k 64 --iter 20 --warmup 10 --cxl
OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --bmm --bsz 3072 --m 1 --n 128 --k 256 --iter 20 --warmup 10 --cxl
OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --bmm --bsz 3072 --m 1 --n 128 --k 1024 --iter 20 --warmup 10 --cxl

# GEMM

echo "GEMM DDR"

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --m 2048 --n 12288 --k 12288 --iter 20 --warmup 10
OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --m 8192 --n 12288 --k 12288 --iter 20 --warmup 10
OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --m 32768 --n 12288 --k 12288 --iter 20 --warmup 10

echo "GEMM CXL"

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --m 2048 --n 12288 --k 12288 --iter 20 --warmup 10 --cxl
OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --m 8192 --n 12288 --k 12288 --iter 20 --warmup 10 --cxl
OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --m 32768 --n 12288 --k 12288 --iter 20 --warmup 10 --cxl