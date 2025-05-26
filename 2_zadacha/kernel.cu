#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>

void blurHost(const unsigned char* src, unsigned char* dst, int shirina, int visota) {
    for (int yy = 0; yy < visota; ++yy)
        for (int xx = 0; xx < shirina; ++xx) {
            int summ = 0, cnt = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    int nx = xx + dx, ny = yy + dy; // соседний пиксель
                    if (nx >= 0 && nx < shirina && ny >= 0 && ny < visota) { // проверка выхода за границы
                        summ += src[ny * shirina + nx];
                        ++cnt;
                    }
                }
            dst[yy * shirina + xx] = summ / cnt;
        }
}

__global__ void blurCuda(const unsigned char* src, unsigned char* dst, int shirina, int visota) {
    int xx = blockIdx.x * blockDim.x + threadIdx.x;
    int yy = blockIdx.y * blockDim.y + threadIdx.y;
    if (xx < shirina && yy < visota) {
        int summ = 0, cnt = 0;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = xx + dx, ny = yy + dy;
                if (nx >= 0 && nx < shirina && ny >= 0 && ny < visota) {
                    summ += src[ny * shirina + nx];
                    ++cnt;
                }
            }
        dst[yy * shirina + xx] = summ / cnt;
    }
}

int main() {
    int shirina = 1024, visota = 1024;
    size_t razmer = shirina * visota;

    unsigned char* imgVhod = new unsigned char[razmer];
    unsigned char* imgVyhodCPU = new unsigned char[razmer];
    unsigned char* imgVyhodGPU = new unsigned char[razmer];


    for (size_t i = 0; i < razmer; ++i)
        imgVhod[i] = rand() % 256;
    auto tStart = std::chrono::high_resolution_clock::now();
    blurHost(imgVhod, imgVyhodCPU, shirina, visota);
    auto tEnd = std::chrono::high_resolution_clock::now();
    std::cout << "CPU: "
        << std::chrono::duration<double, std::milli>(tEnd - tStart).count()
        << " ms\n";


    unsigned char* d_vhod, * d_vyhod;
    cudaMalloc(&d_vhod, razmer);
    cudaMalloc(&d_vyhod, razmer);
    cudaMemcpy(d_vhod, imgVhod, razmer, cudaMemcpyHostToDevice);


    dim3 thPerBlk(16, 16);
    dim3 numBlk((shirina + 15) / 16, (visota + 15) / 16);

    tStart = std::chrono::high_resolution_clock::now();
    blurCuda << <numBlk, thPerBlk >> > (d_vhod, d_vyhod, shirina, visota);
    cudaDeviceSynchronize();
    tEnd = std::chrono::high_resolution_clock::now();
    std::cout << "GPU: "
        << std::chrono::duration<double, std::milli>(tEnd - tStart).count()
        << " ms\n";

    cudaMemcpy(imgVyhodGPU, d_vyhod, razmer, cudaMemcpyDeviceToHost);
  

    cudaFree(d_vhod); cudaFree(d_vyhod);
    delete[] imgVhod; delete[] imgVyhodCPU; delete[] imgVyhodGPU;
    return 0;
}
