import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

device = torch.device('cuda')
print(f"FP4 - Using device: {device}")

class PolyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.net(x)

def target_poly(x):
    return x**3 - 2*x**2 + x + 1

# FP4 simulation using quantization
def quantize_fp4(tensor):
    # Simulate FP4 by aggressive quantization
    scale = tensor.abs().max() / 7.0  # 4-bit range: -7 to 7
    quantized = torch.round(tensor / scale * 7.0).clamp(-7, 7)
    return quantized * scale / 7.0

torch.manual_seed(42)
x_range = torch.linspace(-2, 2, 200).unsqueeze(1).to(device)
y_true = target_poly(x_range)

model = PolyNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Lower LR for stability
criterion = nn.MSELoss()

frames = []
epochs = 1000
save_every = 50

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(6, 4), dpi=80)

for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Quantize weights to simulate FP4
    for param in model.parameters():
        param.data = quantize_fp4(param.data)
    
    y_pred = model(x_range)
    loss = criterion(y_pred, y_true)
    loss.backward()
    optimizer.step()
    
    if epoch % save_every == 0:
        with torch.no_grad():
            y_pred_cpu = y_pred.cpu().numpy()
            x_cpu = x_range.cpu().numpy()
            y_true_cpu = y_true.cpu().numpy()
            
            ax.clear()
            ax.plot(x_cpu, y_true_cpu, 'g-', linewidth=2, label='Target')
            ax.plot(x_cpu, y_pred_cpu, 'r--', linewidth=2, label='FP4')
            ax.set_xlim(-2, 2)
            ax.set_ylim(-5, 5)
            ax.set_title(f'FP4 - Epoch {epoch}, Loss: {loss.item():.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', facecolor='black')
            buf.seek(0)
            frames.append(Image.open(buf).copy())
            buf.close()

plt.close()
frames[0].save('poly_fp4.gif', save_all=True, append_images=frames[1:], duration=200, loop=0)
print(f"FP4 complete! Final loss: {loss.item():.6f}")