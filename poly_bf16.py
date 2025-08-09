from poly_base import train_and_visualize
import torch
# %% [markdown]
print("BF16 PyTorch - Using actual bfloat16 precision")
# %% [markdown]
# # This is a markdown cell
# You can write **formatted** text here

# %%
if __name__ == "__main__":
    train_and_visualize("BF16", torch.bfloat16, lr=0.012, epochs=1500)

# %% [markdown]
"""
# Markdown title
Some text
"""
