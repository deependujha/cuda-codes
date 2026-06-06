import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def launch_add(x, y):
    output = torch.empty_like(x)
    n_elements = x.numel()
    block_size = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Warm up to bypass JIT compilation timing during actual profiling
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=block_size)
    torch.cuda.synchronize()
    
    # This is the actual execution we want to look at
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=block_size)
    torch.cuda.synchronize()
    return output

if __name__ == "__main__":
    size = 1024 * 1024
    x = torch.ones(size, device='cuda')
    y = torch.ones(size, device='cuda')
    launch_add(x, y)
