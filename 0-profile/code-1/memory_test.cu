#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>
#include <stdio.h>

#define MATRIX_SIZE 128

// BAD: Each thread reads with a stride of MATRIX_SIZE
__global__ void uncoalesced_read(float *out, float *in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Stride access patterns force multiple memory requests
        out[idx] = in[(idx * 32) % N]; 
    }
}

// GOOD: Consecutive threads read consecutive memory addresses
__global__ void coalesced_read(float *out, float *in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = in[idx];
    }
}

int main() {
    int N = MATRIX_SIZE * MATRIX_SIZE;
    size_t bytes = N * sizeof(float);

    float *h_in = (float*)malloc(bytes);
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Profile Uncoalesced
    nvtxRangePushA("Uncoalesced Kernel Run");
    uncoalesced_read<<<blocks, threads>>>(d_out, d_in, N);
    cudaDeviceSynchronize();
    nvtxRangePop();

    // Profile Coalesced
    nvtxRangePushA("Coalesced Kernel Run");
    coalesced_read<<<blocks, threads>>>(d_out, d_in, N);
    cudaDeviceSynchronize();

    nvtxRangePop();

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    return 0;
}
