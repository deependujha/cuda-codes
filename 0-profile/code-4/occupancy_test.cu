#include <cuda_runtime.h>
#include <stdio.h>

// A kernel that forces the compiler to use a ton of registers 
// by declaring many variables and doing dependent math.
__global__ void high_register_kernel(float *out, float *in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float a1 = in[idx];
        float a2 = a1 * 2.0f; float a3 = a2 + 3.0f; float a4 = a3 / 4.0f;
        float a5 = a4 - 5.0f; float a6 = a5 * 6.0f; float a7 = a6 + 7.0f;
        float a8 = a7 / 8.0f; float a9 = a8 - 9.0f; float a10 = a9 * 10.0f;
        
        // Volatile array forces registers to stay alive and not be optimized away
        volatile float local_arr[10];
        local_arr[0] = a1; local_arr[1] = a2; local_arr[2] = a3; local_arr[3] = a4;
        local_arr[4] = a5; local_arr[5] = a6; local_arr[6] = a7; local_arr[7] = a8;
        local_arr[8] = a9; local_arr[9] = a10;

        out[idx] = local_arr[idx % 10] + a10;
    }
}

int main() {
    int N = 1 << 20;

    size_t bytes = N * sizeof(float);
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    // Launching with large block size to maximize resource pressure per SM
    // int threads = 256;
    int threads = 1024;
    int blocks = (N + threads - 1) / threads;

    high_register_kernel<<<blocks, threads>>>(d_out, d_in, N);
    cudaDeviceSynchronize();

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
