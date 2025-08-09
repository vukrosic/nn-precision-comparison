from poly_base import train_and_visualize
import torch

print("FP16 PyTorch - Using actual float16 precision")

if __name__ == "__main__":
    train_and_visualize("FP16", torch.float16, lr=0.008, epochs=1500)