#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>
#include <stdio.h>

#define VECTOR_SIZE (1 << 24) // Large vector
#define NUM_STREAMS 4

__global__ void intense_math_kernel(float *out, float *in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Just some math to keep the GPU busy for a moment
        float val = in[idx];
        for(int i = 0; i < 200; i++) {
            val = rsqrtf(val + 1.0f) + 0.5f;
        }
        out[idx] = val;
    }
}

int main() {
    size_t bytes = VECTOR_SIZE * sizeof(float);
    
    // Page-locked (pinned) host memory is REQUIRED for async memcpys
    float *h_in, *h_out;
    cudaHostAlloc(&h_in, bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_out, bytes, cudaHostAllocDefault);
    
    // Initialize host data
    for(int i = 0; i < VECTOR_SIZE; i++) h_in[i] = 1.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    int threads = 256;
    int blocks = (VECTOR_SIZE + threads - 1) / threads;

    // ==========================================
    // APPROACH 1: Synchronous (The Default Way)
    // ==========================================
    nvtxRangePushA("APPROACH_1_SYNCHRONOUS");
    
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    intense_math_kernel<<<blocks, threads>>>(d_out, d_in, VECTOR_SIZE);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    nvtxRangePop();

    // ==========================================
    // APPROACH 2: Asynchronous (The Streams Way)
    // ==========================================
    nvtxRangePushA("APPROACH_2_STREAMS");

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) cudaStreamCreate(&streams[i]);

    int chunk_size = VECTOR_SIZE / NUM_STREAMS;
    size_t chunk_bytes = chunk_size * sizeof(float);

    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * chunk_size;
        
        // Use cudaMemcpyAsync and pass the specific stream
        cudaMemcpyAsync(d_in + offset, h_in + offset, chunk_bytes, cudaMemcpyHostToDevice, streams[i]);
        
        int intense_blocks = (chunk_size + threads - 1) / threads;
        intense_math_kernel<<<intense_blocks, threads, 0, streams[i]>>>(d_out + offset, d_in + offset, chunk_size);
        
        cudaMemcpyAsync(h_out + offset, d_out + offset, chunk_bytes, cudaMemcpyDeviceToHost, streams[i]);
    }

    // Wait for all streams to finish
    cudaDeviceSynchronize();
    nvtxRangePop();

    // Cleanup
    for (int i = 0; i < NUM_STREAMS; i++) cudaStreamDestroy(streams[i]);
    cudaFree(d_in); cudaFree(d_out);
    cudaFreeHost(h_in); cudaFreeHost(h_out);
    return 0;
}
