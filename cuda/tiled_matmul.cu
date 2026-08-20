#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_SIZE 16
#define TILE_SIZE_2D 64
#define N 1024
#define TM_1d 8
#define TM 4
#define TN 4

__global__ void matmul_naive(float* A, float* B, float* C, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n)
    {
        float tmp = 0.0f;

        for (int k = 0; k < n; ++k)
            tmp += A[row * n + k] * B[k * n + col];

        C[row * n + col] = tmp;
    }
}



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
    __shared__ float tileA[TILE_SIZE * TM_1d][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int blockRowStart = blockIdx.y * (TILE_SIZE * TM_1d);
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float tmp[TM_1d] = {0.0f};

    int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++)
    {
        int bRow = t * TILE_SIZE + threadIdx.y;
        tileB[threadIdx.y][threadIdx.x] =
            (bRow < n && col < n) ? B[bRow * n + col] : 0.0f;

        for (int i = 0; i < TM_1d; i++)
        {
            int aRow = blockRowStart + i * TILE_SIZE + threadIdx.y;
            int aCol = t * TILE_SIZE + threadIdx.x;
            tileA[i * TILE_SIZE + threadIdx.y][threadIdx.x] =
                (aRow < n && aCol < n) ? A[aRow * n + aCol] : 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
            for (int i = 0; i < TM_1d; i++)
                tmp[i] += tileA[i * TILE_SIZE + threadIdx.y][k]
                         * tileB[k][threadIdx.x];

        __syncthreads();
    }

    for (int i = 0; i < TM_1d; i++)
    {
        int row = blockRowStart + i * TILE_SIZE + threadIdx.y;
        if (row < n && col < n)
            C[row * n + col] = tmp[i];
    }
}


//Kerenel 3: 2D Tiling
//===============================================================
__global__ void matmul_2d_tiled(float* A, float* B, float* C, int n){
  __shared__ float tileA[TILE_SIZE_2D][TILE_SIZE_2D];
  __shared__ float tileB[TILE_SIZE_2D][TILE_SIZE_2D];

  int blockRowStart = blockIdx.y * TILE_SIZE_2D;
  int blockColStart = blockIdx.x * TILE_SIZE_2D;

  //flat thread id, used only for cooperative tile loading
  int threadsPerBlock = blockDim.x * blockDim.y;
  int threadFlat = threadIdx.y * blockDim.x + threadIdx.x;

  //This thread's output: a 4x4 block, kept entirely in registers
  float results[TM][TN] = {{0.0f}};

  float colOfA[TM]; //TM values pulled from tileA's column
  float rowOfB[TN]; //TN values pulled from tileB's row'

  int numTiles = (n + TILE_SIZE_2D - 1) / TILE_SIZE_2D;

  for(int t = 0; t < numTiles; t ++){
    int elemsPerTile = TILE_SIZE_2D * TILE_SIZE_2D;
    for(int idx = threadFlat; idx < elemsPerTile; idx += threadsPerBlock){
      int r = idx / TILE_SIZE_2D;
      int c = idx % TILE_SIZE_2D;

      int aRow = blockRowStart + r;
      int aCol = t* TILE_SIZE_2D + c;
      tileA[r][c] = (aRow < n && aCol < n) ? A[aRow * n + aCol] : 0.0f;

      int bRow = t * TILE_SIZE_2D + r;
      int bCol = blockColStart + c;
      tileB[r][c] = (bRow < n && bCol < n) ? B[bRow * n + bCol] : 0.0f;
    }
    __syncthreads();

  for(int k = 0; k < TILE_SIZE_2D; k++){
    //pull this thread's column slice of tileA into registers
    for(int i = 0; i < TM; i++){
      colOfA[i] = tileA[threadIdx.y * TM + i][k];
    }

    //pull this thread's row slice of tileB into registers
    for(int j = 0; j < TN; j++){
      rowOfB[j] = tileB[k][threadIdx.x * TN + j];
    }

    //outer product: 4x4 = 16 FMAs, purely register to register
    for(int i = 0; i < TM; i++){
      for(int j = 0; j < TN; j++){
        results[i][j] += colOfA[i] * rowOfB[j];
      }
    }
  }

  __syncthreads();
  }

  //Write this thread's 8x8 output back to c
  for(int i =0; i < TM; i++){
    for(int j = 0; j < TN; j++){
      int row = blockRowStart + threadIdx.y * TM + i;
      int col = blockColStart + threadIdx.x * TN + j;
      if(row < n && col < n){
        C[row * n + col] = results[i][j];
      }
    }
  }

}



// Kernel 4: Vectorised Loads (float4)
// ============================================================
__global__ void matmul_vec4(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int n)
{
    __shared__ float tileA[TILE_SIZE_2D][TILE_SIZE_2D];
    __shared__ float tileB[TILE_SIZE_2D][TILE_SIZE_2D];

    int blockRowStart = blockIdx.y * TILE_SIZE_2D;
    int blockColStart = blockIdx.x * TILE_SIZE_2D;

    int threadsPerBlock = blockDim.x * blockDim.y;
    int threadFlat = threadIdx.y * blockDim.x + threadIdx.x;

    float tmp[TM][TN] = {{0.0f}};
    float regA[TM];
    float regB[TN];

    int numTiles  = (n + TILE_SIZE_2D - 1) / TILE_SIZE_2D;
    int vec4PerTile = (TILE_SIZE_2D * TILE_SIZE_2D) / 4;

    for (int t = 0; t < numTiles; t++)
    {
        float4* tileA_vec = reinterpret_cast<float4*>(tileA);
        const float4* A_vec = reinterpret_cast<const float4*>(A);

        for (int idx = threadFlat; idx < vec4PerTile; idx += threadsPerBlock)
        {
            int flatFloat = idx * 4;
            int r = flatFloat / TILE_SIZE_2D;
            int c = flatFloat % TILE_SIZE_2D;

            int aRow = blockRowStart + r;
            int aCol = t * TILE_SIZE_2D + c;

            if (aRow < n && aCol + 3 < n)
                tileA_vec[idx] = A_vec[aRow * (n / 4) + aCol / 4];
            else
            {
                tileA[r][c + 0] = (aRow < n && aCol + 0 < n) ? A[aRow * n + aCol + 0] : 0.0f;
                tileA[r][c + 1] = (aRow < n && aCol + 1 < n) ? A[aRow * n + aCol + 1] : 0.0f;
                tileA[r][c + 2] = (aRow < n && aCol + 2 < n) ? A[aRow * n + aCol + 2] : 0.0f;
                tileA[r][c + 3] = (aRow < n && aCol + 3 < n) ? A[aRow * n + aCol + 3] : 0.0f;
            }
        }

        float4* tileB_vec = reinterpret_cast<float4*>(tileB);
        const float4* B_vec = reinterpret_cast<const float4*>(B);

        for (int idx = threadFlat; idx < vec4PerTile; idx += threadsPerBlock)
        {
            int flatFloat = idx * 4;
            int r = flatFloat / TILE_SIZE_2D;
            int c = flatFloat % TILE_SIZE_2D;

            int bRow = t * TILE_SIZE_2D + r;
            int bCol = blockColStart + c;

            if (bRow < n && bCol + 3 < n)
                tileB_vec[idx] = B_vec[bRow * (n / 4) + bCol / 4];
            else
            {
                tileB[r][c + 0] = (bRow < n && bCol + 0 < n) ? B[bRow * n + bCol + 0] : 0.0f;
                tileB[r][c + 1] = (bRow < n && bCol + 1 < n) ? B[bRow * n + bCol + 1] : 0.0f;
                tileB[r][c + 2] = (bRow < n && bCol + 2 < n) ? B[bRow * n + bCol + 2] : 0.0f;
                tileB[r][c + 3] = (bRow < n && bCol + 3 < n) ? B[bRow * n + bCol + 3] : 0.0f;
            }
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE_2D; k++)
        {
            for (int i = 0; i < TM; i++)
                regA[i] = tileA[threadIdx.y * TM + i][k];
            float4 b4 = reinterpret_cast<const float4*>(&tileB[k][threadIdx.x * TN])[0];
            regB[0] = b4.x;
            regB[1] = b4.y;
            regB[2] = b4.z;
            regB[3] = b4.w;

            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    tmp[i][j] += regA[i] * regB[j];
        }

        __syncthreads();
    }

    for (int i = 0; i < TM; ++i)
{
    const int row = blockRowStart + threadIdx.y * TM + i;
    const int col = blockColStart + threadIdx.x * TN;

    if (row < n && col + 3 < n)
    {
        reinterpret_cast<float4*>(&C[row * n + col])[0] =
            make_float4(tmp[i][0], tmp[i][1], tmp[i][2], tmp[i][3]);
    }
    else
    {
        for (int j = 0; j < TN; ++j)
            if (row < n && col + j < n)
                C[row * n + col + j] = tmp[i][j];
    }
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

    double peak = 2.0 * 2048 * 2.100;  // 8601.6 GFLOPS, RTX 3050 Laptop
    int runs = 100;
    float ms = 0.0f;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // -- Kernel 0: Naive ------------------------------------------
    {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (N + 15) / 16);

    matmul_naive<<<blocks, threads>>>(d_A, d_B, d_C, N);  // Warm-up
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < runs; ++i)
        matmul_naive<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&ms, start, end);
    ms /= runs;

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    printf("\n=== Kernel 0: Naive ===\n");
    verify(A, B, C, N, "naive");

    double gf = (2.0 * N * N * N) / (ms * 1e-3) / 1e9;
    printf("  Time: %.3f ms | %.2f GFLOPS | %.2f%% peak\n",
           ms, gf, gf / peak * 100);
    }



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
    ms /= runs;

    // Copy result back
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify correctness
    printf("\n=== Basic Tiled ===\n");
    verify(A, B, C, N, "basic");
    double gf = (2.0 * N * N * N) / (ms * 1e-3) / 1e9;
    printf("  Time: %.3f ms | %.2f GFLOPS | %.2f%% peak\n", ms, gf, gf / peak * 100);
    }


    // -- Kernel 2: 1D Tiled -------------------------------------
    {
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                    (N + TILE_SIZE * TM_1d - 1) / (TILE_SIZE * TM_1d));

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
    printf("\n=== 1D Tiled (TM=%d) ===\n", TM_1d);
    verify(A, B, C, N, "1d-tiled");
    double gf = (2.0 * N * N * N) / (ms * 1e-3) / 1e9;
    printf("  Time: %.3f ms | %.2f GFLOPS | %.2f%% peak\n", ms, gf, gf / peak * 100);
    }


    // -- Kernel 3: 2D Tiled --------------------------------
    {
        dim3 threads(TILE_SIZE_2D / TN, TILE_SIZE_2D / TM);
        dim3 blocks((N + TILE_SIZE_2D - 1) / TILE_SIZE_2D,
                       (N + TILE_SIZE_2D - 1) / TILE_SIZE_2D);

        matmul_2d_tiled<<<blocks, threads>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < runs; i++)
            matmul_2d_tiled<<<blocks, threads>>>(d_A, d_B, d_C, N);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&ms, start, end);
        ms /= runs;

        cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
        printf("\n=== Kernel 3: 2D Tiled (TM=%d, TN=%d, TILE=%d) ===\n",
               TM, TN, TILE_SIZE_2D);
        verify(A, B, C, N, "2d-tiled");
        double gf = (2.0 * N * N * N) / (ms * 1e-3) / 1e9;
        printf("  Time: %.3f ms | %.2f GFLOPS | %.2f%% peak\n", ms, gf, gf / peak * 100);
    }


    // -- Kernel 4: Vectorised Loads (float4) ----------------------
    {
        dim3 k4_threads(TILE_SIZE_2D / TN, TILE_SIZE_2D / TM);
        dim3 k4_blocks((N + TILE_SIZE_2D - 1) / TILE_SIZE_2D,
                       (N + TILE_SIZE_2D - 1) / TILE_SIZE_2D);

        matmul_vec4<<<k4_blocks, k4_threads>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < runs; i++)
            matmul_vec4<<<k4_blocks, k4_threads>>>(d_A, d_B, d_C, N);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&ms, start, end);
        ms /= runs;

        cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
        printf("\n=== Kernel 4: Vectorised Loads (float4) ===\n");
        verify(A, B, C, N, "vec4");
        double gf = (2.0 * N * N * N) / (ms * 1e-3) / 1e9;
        printf("  Time: %.3f ms | %.2f GFLOPS | %.2f%% peak\n", ms, gf, gf / peak * 100);
    }


    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C);

    return 0;
}
