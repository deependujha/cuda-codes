# Commands

```bash
# 1. Compile
nvcc -O3 -lineinfo occupancy_test.cu -o occupancy_test

# 2. Profile with nsys
ncu --set full -o occupancy_study ./occupancy_test
```
