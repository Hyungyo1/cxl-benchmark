# cxl-benchmark

For GEMV microbenchmark, use --bmm, --amx, --bsz=3072, --m=1, --n=128, --k=64 (and 256, 1024)
For GEMM microbenchmark, use --amx, --m=2048 (and 8192, 32768), --n=12288, --k=12288

--cxl: use cxl to store one tensor
--amx: use AMX
