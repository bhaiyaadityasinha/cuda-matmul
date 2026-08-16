%%writefile cuda/tiled_matmul.cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_SIZE 16
#define N 1024
#define TM 8
#define TN 8


//Kernel 1: Shared memory caching
//===============================================================
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


// Kernel 2: 1D Tiling
//=================================================================
__global__ void matmul_1d_tiled(float* A, float* B, float* C, int n)
{
    __shared__ float tileA[TILE_SIZE * TM][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int blockRowStart = blockIdx.y * (TILE_SIZE * TM);
    int col           = blockIdx.x * TILE_SIZE + threadIdx.x;

    float tmp[TM] = {0.0f};

    int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++)
    {
        int bRow = t * TILE_SIZE + threadIdx.y;
        tileB[threadIdx.y][threadIdx.x] =
            (bRow < n && col < n) ? B[bRow * n + col] : 0.0f;

        for (int i = 0; i < TM; i++)
        {
            int aRow = blockRowStart + i * TILE_SIZE + threadIdx.y;
            int aCol = t * TILE_SIZE + threadIdx.x;
            tileA[i * TILE_SIZE + threadIdx.y][threadIdx.x] =
                (aRow < n && aCol < n) ? A[aRow * n + aCol] : 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
            for (int i = 0; i < TM; i++)
                tmp[i] += tileA[i * TILE_SIZE + threadIdx.y][k]
                         * tileB[k][threadIdx.x];

        __syncthreads();
    }

    for (int i = 0; i < TM; i++)
    {
        int row = blockRowStart + i * TILE_SIZE + threadIdx.y;
        if (row < n && col < n)
            C[row * n + col] = tmp[i];
    }
}


// Verify
// ===============================================================
static void verify(const float* A, const float* B, const float* C,
                   int n, const char* label)
{
    const int CHECKS = 1000;
    bool correct = true;
    for (int i = 0; i < CHECKS && correct; i++)
    {
        int row = rand() % n, col = rand() % n;
        float expected = 0.0f;
        for (int k = 0; k < n; k++)
            expected += A[row * n + k] * B[k * n + col];
        if (fabsf(C[row * n + col] - expected) > 1e-3f)
        {
            printf("  [%s] MISMATCH at C[%d][%d]: got %.6f expected %.6f\n",
                   label, row, col, C[row * n + col], expected);
            correct = false;
        }
    }
    printf("  [%s] Correct: %s (%d samples)\n",
           label, correct ? "True" : "False", CHECKS);
}


int main()
{
    srand(42);
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

    double peak = 8100.0;
    int runs = 10;
    float ms = 0.0f;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);


    //--Kernel 1: Basic Tiled-------------------------------
    {
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (N + TILE_SIZE - 1) / TILE_SIZE);

    matmul_tiled<<<blocks, threads>>>(d_A, d_B, d_C, N);  // Warm-up
    cudaDeviceSynchronize();

    cudaEventRecord(start);

    for (int i = 0; i < runs; i++)
        matmul_tiled<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&ms, start, end);
    ms /= 10;

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    printf("\n=== Basic Tiled ===\n");
    verify(A, B, C, N, "basic");
    double gf = (2.0 * N * N * N) / (ms * 1e-3) / 1e9;
    printf("  Time: %.3f ms | %.2f GFLOPS | %.2f%% peak\n", ms, gf, gf / peak * 100);
    }


    // -- Kernel 2: 1D Tiled -------------------------------------
    {
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                    (N + TILE_SIZE * TM - 1) / (TILE_SIZE * TM));

    matmul_1d_tiled<<<blocks, threads>>>(d_A, d_B, d_C, N); // warm-up
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < runs; i++)
    matmul_1d_tiled<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms, start, end);
    ms /= runs;

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    printf("\n=== 1D Tiled (TM=%d) ===\n", TM);
    verify(A, B, C, N, "1d-tiled");
    double gf = (2.0 * N * N * N) / (ms * 1e-3) / 1e9;
    printf("  Time: %.3f ms | %.2f GFLOPS | %.2f%% peak\n", ms, gf, gf / peak * 100);
    }

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
