#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>

using namespace std;

void matmul_naive_cpu(const vector<float>& A,
                  const vector<float>& B,
                  vector<float>& C,
                  int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i * N + j] += A[i * N + k] * B[k * N + j];
}

void matmul_cache_friendly_cpu(const vector<float>& A,
                           const vector<float>& B,
                           vector<float>& C,
                           int N)
{
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++)
            for (int j = 0; j < N; j++)
                C[i * N + j] += A[i * N + k] * B[k * N + j];
}

int main()
{
    constexpr int N = 1024;
    constexpr int runs = 5;

    // Same random initialization pattern as CUDA benchmarks
    srand(42);

    vector<float> A(N * N);
    vector<float> B(N * N);
    vector<float> C(N * N, 0.0f);

    for (int i = 0; i < N * N; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    // -- Naive ---------------------------------
    matmul_naive_cpu(A, B, C, N);
    fill(C.begin(), C.end(), 0.0f);

    double total_ms = 0.0;

    for (int r = 0; r < runs; r++) {
        fill(C.begin(), C.end(), 0.0f);

        auto start = chrono::high_resolution_clock::now();

        matmul_naive_cpu(A, B, C, N);

        auto end = chrono::high_resolution_clock::now();

        total_ms += chrono::duration<double, milli>(end - start).count();
    }

    double naive_ms = total_ms / runs;
    double naive_gflops =
        (2.0 * N * N * N) / (naive_ms * 1e-3) / 1e9;

    cout << "  Naive CPU:          "
         << naive_ms << " ms | "
         << naive_gflops << " GFLOPS\n";


    // -- Cache-friendly ------------------------------------
    matmul_cache_friendly_cpu(A, B, C, N);
    fill(C.begin(), C.end(), 0.0f);

    total_ms = 0.0;

    for (int r = 0; r < runs; r++) {
        fill(C.begin(), C.end(), 0.0f);

        auto start = chrono::high_resolution_clock::now();

        matmul_cache_friendly_cpu(A, B, C, N);

        auto end = chrono::high_resolution_clock::now();

        total_ms += chrono::duration<double, milli>(end - start).count();
    }

    double cache_ms = total_ms / runs;
    double cache_gflops =
        (2.0 * N * N * N) / (cache_ms * 1e-3) / 1e9;

    cout<< "  Cache-friendly CPU: "
        <<cache_ms<< " ms | "
        <<cache_gflops<< " GFLOPS\n";

    cout<< "  Cache speedup vs naive: "
        << naive_ms / cache_ms<< "x\n";

    return 0;
}
