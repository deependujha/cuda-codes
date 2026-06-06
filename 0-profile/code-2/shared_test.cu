#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 32

// BAD: Stride causes multiple threads to hit the exact same shared memory banks
__global__ void conflict_kernel(float *out, float *in) {
    __shared__ float s_data[BLOCK_SIZE * BLOCK_SIZE];

    int tid = threadIdx.x;
    // Load data cleanly
    s_data[tid] = in[tid];
    __syncthreads();

    // Read with a stride of 32 -> Bank Conflict!
    // Thread 0 reads s_data[0] (Bank 0)
    // Thread 1 reads s_data[32] (Bank 0)
    // Thread 2 reads s_data[64] (Bank 0)... 32-way conflict!
    out[tid] = s_data[tid * BLOCK_SIZE];

    // ...

}

// BAD: Stride causes multiple threads to hit the exact same shared memory banks
__global__ void conflict_kernel_2(float *out, float *in) {
    __shared__ float s_data[BLOCK_SIZE * BLOCK_SIZE];
    
    int tid = threadIdx.x;
    // Load data cleanly
    s_data[tid] = in[tid];
    __syncthreads();
    
    // Read with a stride of 32 -> Bank Conflict!
    // Thread 0 reads s_data[0] (Bank 0)
    // Thread 1 reads s_data[32] (Bank 0)
    // Thread 2 reads s_data[64] (Bank 0)... 32-way conflict!
    out[tid] = s_data[tid * 2];
}

// GOOD: Linear access utilizes parallel banks perfectly
__global__ void clean_kernel(float *out, float *in) {
    __shared__ float s_data[BLOCK_SIZE * BLOCK_SIZE];
    
    int tid = threadIdx.x;
    s_data[tid] = in[tid];
    __syncthreads();
    
    // Sequential read -> No conflicts
    out[tid] = s_data[tid];
}

int main() {
    size_t bytes = BLOCK_SIZE * BLOCK_SIZE * sizeof(float);
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    conflict_kernel<<<1, BLOCK_SIZE>>>(d_out, d_in);
    cudaDeviceSynchronize();

    conflict_kernel_2<<<1, BLOCK_SIZE>>>(d_out, d_in);
    cudaDeviceSynchronize();

    clean_kernel<<<1, BLOCK_SIZE>>>(d_out, d_in);
    cudaDeviceSynchronize();

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}