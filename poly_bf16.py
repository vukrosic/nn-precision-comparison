import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

print("BF16 NumPy - Simple implementation")

# BF16 quantization: 16 bits with 1 sign, 8 exponent, 7 mantissa bits
def quantize_bf16(x):
    # Simulate BF16 by truncating mantissa bits
    # BF16 has same exponent range as FP32 but fewer mantissa bits
    # We approximate this by rounding to fewer significant digits
    scale = 2**7  # Approximate BF16 precision
    return np.round(x * scale) / scale

def relu(x):
    return np.maximum(0, x)

def target_poly(x):
    return x**3 - 2*x**2 + x + 1

# Simple 3-layer network
class SimpleNet:
    def __init__(self):
        # Initialize weights
        self.w1 = np.random.normal(0, 0.1, (1, 32))
        self.b1 = np.zeros((1, 32))
        self.w2 = np.random.normal(0, 0.1, (32, 32))
        self.b2 = np.zeros((1, 32))
        self.w3 = np.random.normal(0, 0.1, (32, 1))
        self.b3 = np.zeros((1, 1))
    
    def forward(self, x):
        # Quantize weights for BF16 simulation
        w1_q = quantize_bf16(self.w1)
        w2_q = quantize_bf16(self.w2)
        w3_q = quantize_bf16(self.w3)
        
        z1 = x @ w1_q + self.b1
        a1 = relu(z1)
        z2 = a1 @ w2_q + self.b2
        a2 = relu(z2)
        z3 = a2 @ w3_q + self.b3
        return z3, (z1, a1, z2, a2)
    
    def backward(self, x, y_true, y_pred, cache, lr=0.01):
        z1, a1, z2, a2 = cache
        m = x.shape[0]
        
        # Output layer gradients
        dz3 = y_pred - y_true
        dw3 = a2.T @ dz3 / m
        db3 = np.mean(dz3, axis=0, keepdims=True)
        
        # Hidden layer 2 gradients
        da2 = dz3 @ quantize_bf16(self.w3).T
        dz2 = da2 * (z2 > 0)  # ReLU derivative
        dw2 = a1.T @ dz2 / m
        db2 = np.mean(dz2, axis=0, keepdims=True)
        
        # Hidden layer 1 gradients
        da1 = dz2 @ quantize_bf16(self.w2).T
        dz1 = da1 * (z1 > 0)  # ReLU derivative
        dw1 = x.T @ dz1 / m
        db1 = np.mean(dz1, axis=0, keepdims=True)
        
        # Update weights
        self.w1 -= lr * dw1
        self.b1 -= lr * db1
        self.w2 -= lr * dw2
        self.b2 -= lr * db2
        self.w3 -= lr * dw3
        self.b3 -= lr * db3

# Setup
np.random.seed(42)
x_range = np.linspace(-2, 2, 200).reshape(-1, 1)
y_true = target_poly(x_range)

model = SimpleNet()
frames = []
epochs = 1000
save_every = 50
lr = 0.01

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(6, 4), dpi=80)

for epoch in range(epochs):
    # Forward pass
    y_pred, cache = model.forward(x_range)
    loss = np.mean((y_pred - y_true)**2)
    
    # Backward pass
    model.backward(x_range, y_true, y_pred, cache, lr)
    
    if epoch % save_every == 0:
        ax.clear()
        ax.plot(x_range, y_true, 'g-', linewidth=2, label='Target')
        ax.plot(x_range, y_pred, 'r--', linewidth=2, label='BF16')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-5, 5)
        ax.set_title(f'BF16 NumPy - Epoch {epoch}, Loss: {loss:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor='black')
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()

plt.close()
frames[0].save('poly_bf16.gif', save_all=True, append_images=frames[1:], duration=200, loop=0)
print(f"BF16 NumPy complete! Final loss: {loss:.6f}")