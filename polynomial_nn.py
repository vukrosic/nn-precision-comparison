import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Simple neural network
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

# Generate polynomial data: y = x^3 - 2x^2 + x + 1
def target_poly(x):
    return x**3 - 2*x**2 + x + 1

# Setup
torch.manual_seed(42)
x_range = torch.linspace(-2, 2, 200).unsqueeze(1).to(device)
y_true = target_poly(x_range)

model = PolyNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.MSELoss()

# Use mixed precision for GPU efficiency
scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

# Training with animation
frames = []
epochs = 1000
save_every = 50  # Save every 50th epoch

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(6, 4), dpi=80)

for epoch in range(epochs):
    optimizer.zero_grad()
    
    if scaler:  # Mixed precision for GPU
        with torch.cuda.amp.autocast():
            y_pred = model(x_range)
            loss = criterion(y_pred, y_true)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:  # Regular precision for CPU
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
            ax.plot(x_cpu, y_pred_cpu, 'r--', linewidth=2, label='Prediction')
            ax.set_xlim(-2, 2)
            ax.set_ylim(-5, 5)
            ax.set_title(f'Epoch {epoch}, Loss: {loss.item():.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Save frame
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', facecolor='black')
            buf.seek(0)
            frames.append(Image.open(buf).copy())
            buf.close()

plt.close()

# Create GIF
frames[0].save('polynomial_training.gif', 
               save_all=True, 
               append_images=frames[1:], 
               duration=200,  # 200ms per frame
               loop=0)

print(f"Training complete! Final loss: {loss.item():.6f}")
print("Animation saved as 'polynomial_training.gif'")