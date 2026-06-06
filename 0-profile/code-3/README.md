# Commands

```bash
# 1. Compile
nvcc -O3 -lineinfo stream_test.cu -o stream_test

# 2. Profile with nsys
nsys profile --trace=cuda,nvtx,osrt --force-overwrite=true --output=streams_study ./stream_test
```
