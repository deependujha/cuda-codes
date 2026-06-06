# Commands

```bash
# Profile the system timeline with nsys
nsys profile --trace=cuda,nvtx,osrt -o triton_timeline python triton_test.py

# Profile the kernel architecture directly with ncu
# Triton mangles kernel names with metadata hashes, so we search by part of the name
ncu --kernel-name regex:"add_kernel" -o triton_kernel python triton_test.py
```
