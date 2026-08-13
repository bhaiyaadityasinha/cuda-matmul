%%writefile cuda/naive_matmul.cu

#include <stdio.h>
#include <stdlib.h>

#define N 1024

__global__ void matmul_naive(float* A, float* B, float* C, int n) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {

        float tmp = 0.0f;

        for (int k = 0; k < n; k++) {
            tmp += A[row * n + k] * B[k * n + col];
        }

        C[row * n + col] = tmp;
    }
}

int main() {
    int size = N * N * sizeof(float);

    float* A = (float*)malloc(size);
    float* B = (float*)malloc(size);
    float* C = (float*)malloc(size);

    for (int i = 0; i < N * N; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (N + 15) / 16);

    matmul_naive<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);

    for (int i = 0; i < 10; i++)
        matmul_naive<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, end);
    ms /= 10;

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

bool correct = true;

for (int i = 0; i < 10; i++) {
    int row = i;
    int col = i;

    float expected = 0.0f;

    for (int k = 0; k < N; k++)
        expected += A[row * N + k] * B[k * N + col];

    if (abs(C[row * N + col] - expected) > 1e-3f) {
        correct = false;
        break;
    }
}

printf("Correct: %s\n", correct ? "True" : "False");

    double gflops = (2.0 * N * N * N) / (ms * 1e-3) / 1e9;

    printf("Naive CUDA: %.3f ms\n", ms);
    printf("Performance: %.2f GFLOPS\n", gflops);
    printf("T4 Peak efficiency: %.2f%%\n", gflops / 8100.0 * 100);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C);

    return 0;
}
