%%writefile cpu/cpu_matmul.cpp

#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

void matmul_naive(const vector<float>& A,
                  const vector<float>& B,
                  vector<float>& C,
                  int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i * N + j] += A[i * N + k] * B[k * N + j];
}

void matmul_cache_friendly(const vector<float>& A,
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
    int N = 1024;

    vector<float> A(N * N, 1.0f);
    vector<float> B(N * N, 1.0f);
    vector<float> C(N * N, 0.0f);

    auto start = chrono::high_resolution_clock::now();

    matmul_naive(A, B, C, N);

    auto end = chrono::high_resolution_clock::now();

    cout << "Naive CPU: "
         << chrono::duration<double, milli>(end - start).count()
         << " ms\n";

    fill(C.begin(), C.end(), 0.0f);

    start = chrono::high_resolution_clock::now();

    matmul_cache_friendly(A, B, C, N);

    end = chrono::high_resolution_clock::now();

    cout << "Cache-friendly CPU: "
         << chrono::duration<double, milli>(end - start).count()
         << " ms\n";
}
