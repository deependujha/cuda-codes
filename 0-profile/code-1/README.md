# Commands

```bash
nvcc -O3 -lineinfo memory_test.cu -o memory_test
```

```bash
# Get the broad timeline with NVTX markers
nsys profile --trace=cuda,nvtx,osrt --force-overwrite=true --output=timeline_study ./memory_test

# Profile the exact hardware metrics of both kernels
ncu --set full -o kernel_study ./memory_test
```
