# CUDA Matrix Multiplication

CUDA-based matrix multiplication implemented from scratch — a tiled shared-memory kernel achieves a **~1300x speedup** over a naive CPU implementation (N=1024, NVIDIA T4).


## Benchmark

| Implementation | N | Time | Speedup vs naive CPU |
|---|---|---|---|
| Naive CPU | 1024 | 4602.65 ms | 1x |
| Cache-friendly CPU | 1024 | 493.722 ms | 9.3x |
| Naive CUDA | 1024 | 5.408 ms | 851x |
| Tiled CUDA (shared memory) | 1024 | 3.552 ms | 1296x |

The tiled kernel reduces global memory traffic by loading tiles of A and B into on-chip shared memory once per block and reusing them across all threads in that block, instead of each thread re-reading from global memory independently.


## Structure

cpu/ — naive and cache-friendly matrix multiplication in C++
cuda/ — naive and shared-memory tiled matrix multiplication in CUDA C++


## Milestones

- Naive and cache-friendly CPU matmul (C++)
- Naive CUDA matmul, verified against NumPy, benchmarked on T4
- Shared-memory tiled CUDA matmul, benchmarked on T4
