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

### Progress 3
- Implemented naive and cache-friendly CPU matrix multiplication in C++
- Compared loop orders and cache locality

### Benchmark
| Implementation | N | Time |
|---|---:|---:|
| Naive CPU | 1024 | 4602.65 ms |
| Cache-friendly CPU | 1024 | 493.722 ms |
| Naive CUDA | 1024 | 21.391 ms |

## Next
- Implement tiled CUDA matrix multiplication using shared memory
