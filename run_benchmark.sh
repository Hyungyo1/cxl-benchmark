############################## 7.b ##############################

# echo "GEMV DDR"

# OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --bmm --bsz 3072 --m 1 --n 128 --k 64 --iter 10 --warmup 5
# OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --bmm --bsz 3072 --m 1 --n 128 --k 256 --iter 10 --warmup 5
# OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --bmm --bsz 3072 --m 1 --n 128 --k 1024 --iter 10 --warmup 5

# echo "GEMV CXL"

# OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --bmm --bsz 3072 --m 1 --n 128 --k 64 --iter 10 --warmup 5 --cxl
# OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --bmm --bsz 3072 --m 1 --n 128 --k 256 --iter 10 --warmup 5 --cxl
# OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --bmm --bsz 3072 --m 1 --n 128 --k 1024 --iter 10 --warmup 5 --cxl

# echo "GEMM DDR"

# OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --m 2048 --n 12288 --k 12288 --iter 10 --warmup 5
# OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --m 8192 --n 12288 --k 12288 --iter 10 --warmup 5
# OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --m 32768 --n 12288 --k 12288 --iter 10 --warmup 5

# echo "GEMM CXL"

# OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --m 2048 --n 12288 --k 12288 --iter 10 --warmup 5 --cxl
# OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --m 8192 --n 12288 --k 12288 --iter 10 --warmup 5 --cxl
# OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --m 32768 --n 12288 --k 12288 --iter 10 --warmup 5 --cxl


############################## 7.c ##############################

# echo "GEMV no transfer"

# OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --bmm --bsz 3072 --m 1 --n 128 --k 64 --iter 10 --warmup 5 --bwshare 0
# OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --bmm --bsz 3072 --m 1 --n 128 --k 256 --iter 10 --warmup 5 --bwshare 0
# OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --bmm --bsz 3072 --m 1 --n 128 --k 1024 --iter 10 --warmup 5 --bwshare 0

# echo "GEMV from DDR"

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --bmm --bsz 3072 --m 1 --n 128 --k 64 --iter 20 --warmup 10 --bwshare 0
OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --bmm --bsz 3072 --m 1 --n 128 --k 64 --iter 20 --warmup 10 --bwshare 1
# OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --bmm --bsz 3072 --m 1 --n 128 --k 256 --iter 10 --warmup 5 --bwshare 1
# OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --bmm --bsz 3072 --m 1 --n 128 --k 512 --iter 10 --warmup 5 --bwshare 1
# OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --bmm --bsz 3072 --m 1 --n 128 --k 1024 --iter 10 --warmup 5 --bwshare 1

# echo "GEMV from CXL"

# OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --bmm --bsz 3072 --m 1 --n 128 --k 64 --iter 10 --warmup 5 --bwshare 2
# OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --bmm --bsz 3072 --m 1 --n 128 --k 256 --iter 10 --warmup 5 --bwshare 2
# OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --bmm --bsz 3072 --m 1 --n 128 --k 512 --iter 10 --warmup 5 --bwshare 2
# OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --bmm --bsz 3072 --m 1 --n 128 --k 1024 --iter 10 --warmup 5 --bwshare 2

# echo "GEMM no transfer"

# OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --m 2048 --n 12288 --k 12288 --iter 10 --warmup 5 --bwshare 0
# OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --m 8192 --n 12288 --k 12288 --iter 10 --warmup 5 --bwshare 0
# OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --m 32768 --n 12288 --k 12288 --iter 10 --warmup 5 --bwshare 0

# echo "GEMM from DDR"

# OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --m 2048 --n 12288 --k 12288 --iter 10 --warmup 5 --bwshare 1
# OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --m 8192 --n 12288 --k 12288 --iter 10 --warmup 5 --bwshare 1
# OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --m 16384 --n 12288 --k 12288 --iter 10 --warmup 5 --bwshare 1
# OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --m 32768 --n 12288 --k 12288 --iter 10 --warmup 5 --bwshare 1

# echo "GEMM from CXL"

# OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --m 2048 --n 12288 --k 12288 --iter 10 --warmup 5 --bwshare 2
# OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --m 8192 --n 12288 --k 12288 --iter 10 --warmup 5 --bwshare 1
# OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --m 16384 --n 12288 --k 12288 --iter 10 --warmup 5 --bwshare 2
# OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --m 32768 --n 12288 --k 12288 --iter 10 --warmup 5 --bwshare 0