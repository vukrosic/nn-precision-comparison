from poly_base import train_and_visualize
import torch

print("FP4 PyTorch - Using actual 4-bit quantization")

if __name__ == "__main__":
    # Use PyTorch's actual 4-bit quantization - no fallbacks
    dtype = torch.quint4x2  # 4-bit quantized type
    train_and_visualize("FP4", dtype, lr=0.05, epochs=1500)