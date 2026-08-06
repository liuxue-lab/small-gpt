from __future__ import annotations

import platform
import sys
import time

import torch


def bytes_to_gib(num_bytes: int) -> float:
    return num_bytes / 1024**3


print("=" * 64)
print("Small GPT environment check")
print("=" * 64)

print(f"Operating system : {platform.platform()}")
print(f"Python version   : {sys.version.split()[0]}")
print(f"Python executable: {sys.executable}")
print(f"PyTorch version  : {torch.__version__}")
print(f"PyTorch CUDA     : {torch.version.cuda}")

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

print(f"CUDA available   : {cuda_available}")
print(f"Selected device  : {device}")

if cuda_available:
    properties = torch.cuda.get_device_properties(0)

    print(f"GPU name         : {properties.name}")
    print(f"GPU memory       : {bytes_to_gib(properties.total_memory):.2f} GiB")
    print(f"BF16 supported   : {torch.cuda.is_bf16_supported()}")
else:
    print("GPU name         : CPU mode")
    print("GPU memory       : not available")
    print("BF16 supported   : not checked")

torch.manual_seed(1337)

if cuda_available:
    torch.cuda.manual_seed_all(1337)

matrix_size = 2048 if cuda_available else 512

left = torch.randn(
    matrix_size,
    matrix_size,
    device=device,
    requires_grad=True,
)

right = torch.randn(
    matrix_size,
    matrix_size,
    device=device,
)

if cuda_available:
    torch.cuda.synchronize()

start_time = time.perf_counter()

output = left @ right
loss = output.float().square().mean()
loss.backward()

if cuda_available:
    torch.cuda.synchronize()

elapsed_time = time.perf_counter() - start_time

assert torch.isfinite(loss), "Loss contains NaN or Inf"
assert left.grad is not None, "Gradient was not created"
assert torch.isfinite(left.grad).all(), "Gradient contains NaN or Inf"

print("-" * 64)
print(f"Matrix size      : {matrix_size} x {matrix_size}")
print(f"Loss             : {loss.item():.6f}")
print(f"Gradient created : {left.grad is not None}")
print(f"Elapsed time     : {elapsed_time:.4f} seconds")
print("=" * 64)
print("Environment check passed.")