# cuda-matmul
CUDA-based matrix multiplication implemented from scratch.

## Development Log
### Progress 1
* Implemented a vector addition kernel

### Progress 2
* Implemented a naive CUDA matrix multiplication kernel
* Each thread computes a single output element
* Verified correctness against NumPy reference implementation
* Benchmarked on an NVIDIA T4 GPU

### Benchmark
**N = 1024**

| Implementation |          Time |
| -------------- | ------------: |
| Naive CUDA     | **21.391 ms** |

### Next Steps
* Implement tiled matrix multiplication using shared memory

