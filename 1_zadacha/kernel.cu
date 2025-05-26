#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

const int M = 512, N = 512, P = 512;

void matrixMultiplyCPU(const float* A, const float* B, float* C, int m, int n, int p) {
    for (int i = 0; i < m * p; ++i) C[i] = 0.0f;
    for (int i = 0; i < m; ++i)
        for (int k = 0; k < n; ++k) {
            float a = A[i * n + k];
            for (int j = 0; j < p; ++j)
                C[i * p + j] += a * B[k * p + j];
        }
}

__global__ void matrixMultiplyCUDA(const float* A, const float* B, float* C, int m, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < p) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k)
            sum += A[row * n + k] * B[k * p + col];
        C[row * p + col] = sum;
    }
}

int main() {
    float* h_A = new float[M * N],
        * h_B = new float[N * P],
        * h_C_cpu = new float[M * P],
        * h_C_gpu = new float[M * P];

    for (int i = 0; i < M * N; ++i) h_A[i] = rand();
    for (int i = 0; i < N * P; ++i) h_B[i] = rand();

    auto cpu_start = std::chrono::high_resolution_clock::now();
    matrixMultiplyCPU(h_A, h_B, h_C_cpu, M, N, P);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    std::cout << "CPU time: " << cpu_ms << " ms\n";
    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, N * P * sizeof(float));
    cudaMalloc(&d_C, M * P * sizeof(float));

    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * P * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16), grid((P + 15) / 16, (M + 15) / 16);

    auto gpu_start = std::chrono::high_resolution_clock::now();
    matrixMultiplyCUDA << <grid, block >> > (d_A, d_B, d_C, M, N, P);
    cudaDeviceSynchronize();
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
    std::cout << "GPU time: " << gpu_ms << " ms\n";

    cudaMemcpy(h_C_gpu, d_C, M * P * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C_cpu; delete[] h_C_gpu;
    return 0;
}
