from poly_base import train_and_visualize
import torch

print("FP8 PyTorch - Using actual float8_e4m3fn precision")

if __name__ == "__main__":
    # Use PyTorch's actual FP8 dtype - no fallbacks
    dtype = torch.float8_e4m3fn
    train_and_visualize("FP8", dtype, lr=0.01, epochs=1500)