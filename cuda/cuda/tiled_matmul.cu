%%writefile cuda/tiled_matmul.cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_SIZE 16
#define N 1024

__global__ void matmul_tiled(float* A, float* B, float* C, int n)
{
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float tmp = 0.0f;

    int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++)
    {
        if (row < n && (t * TILE_SIZE + threadIdx.x) < n)
            tileA[threadIdx.y][threadIdx.x] =
                A[row * n + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < n && (t * TILE_SIZE + threadIdx.y) < n)
            tileB[threadIdx.y][threadIdx.x] =
                B[(t * TILE_SIZE + threadIdx.y) * n + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
            tmp += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = tmp;
}

int main()
{
    int size = N * N * sizeof(float);

    float* A = (float*)malloc(size);
    float* B = (float*)malloc(size);
    float* C = (float*)malloc(size);

    for (int i = 0; i < N * N; i++)
    {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (N + TILE_SIZE - 1) / TILE_SIZE);

    // Warm-up
    matmul_tiled<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);

    for (int i = 0; i < 10; i++)
        matmul_tiled<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, end);
    ms /= 10;

    // Copy result back
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify correctness
    bool correct = true;

    for (int i = 0; i < 10; i++)
    {
        int row = i;
        int col = i;

        float expected = 0.0f;

        for (int k = 0; k < N; k++)
            expected += A[row * N + k] * B[k * N + col];

        if (fabsf(C[row * N + col] - expected) > 1e-3f)
        {
            correct = false;
            break;
        }
    }

    double gflops = (2.0 * N * N * N) / (ms * 1e-3) / 1e9;

    printf("Correct: %s\n", correct ? "True" : "False");
    printf("Tiled CUDA: %.3f ms\n", ms);
    printf("Performance: %.2f GFLOPS\n", gflops);
    printf("T4 Peak efficiency: %.2f%%\n", gflops / 8100.0 * 100);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return 0;
}
