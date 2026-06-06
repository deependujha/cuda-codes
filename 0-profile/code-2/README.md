# Commands

```bash
nvcc -O3 -lineinfo shared_test.cu -o shared_test
```

```bash
# Get the broad timeline with NVTX markers
nsys profile --trace=cuda,nvtx,osrt --force-overwrite=true --output=02_timeline_study ./shared_test

# Profile the exact hardware metrics of both kernels
ncu --set full -o 02_kernel_study ./shared_test
```
