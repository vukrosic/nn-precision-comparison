import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import io

def target_poly(x):
    return x**3 - 2*x**2 + x + 1

class PolyNet(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        # Larger network: 1 -> 64 -> 64 -> 32 -> 1
        self.fc1 = torch.nn.Linear(1, 64, dtype=dtype)
        self.fc2 = torch.nn.Linear(64, 64, dtype=dtype)
        self.fc3 = torch.nn.Linear(64, 32, dtype=dtype)
        self.fc4 = torch.nn.Linear(32, 1, dtype=dtype)
        
        # Initialize weights
        torch.nn.init.normal_(self.fc1.weight, 0, 0.1)
        torch.nn.init.normal_(self.fc2.weight, 0, 0.1)
        torch.nn.init.normal_(self.fc3.weight, 0, 0.1)
        torch.nn.init.normal_(self.fc4.weight, 0, 0.1)
    
    def forward(self, x):
        # Forward pass with swish activation
        x = self.fc1(x)
        x = F.silu(x)  # SiLU is the same as Swish
        x = self.fc2(x)
        x = F.silu(x)
        x = self.fc3(x)
        x = F.silu(x)
        x = self.fc4(x)
        return x
    


def train_and_visualize(precision_name, dtype, lr=0.01, epochs=1500):
    """Shared training and visualization code"""
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    x_range = torch.linspace(-2, 2, 200, dtype=dtype, device=device).reshape(-1, 1)
    y_true = target_poly(x_range)
    
    model = PolyNet(dtype).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    
    frames = []
    save_every = 75  # Save fewer frames for smoother animation
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6, 4), dpi=80)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        y_pred = model(x_range)
        loss = criterion(y_pred, y_true)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if epoch % save_every == 0:
            ax.clear()
            x_np = x_range.cpu().numpy()
            y_true_np = y_true.detach().cpu().numpy()
            y_pred_np = y_pred.detach().cpu().numpy()
            
            ax.plot(x_np, y_true_np, 'g-', linewidth=2, label='Target')
            ax.plot(x_np, y_pred_np, 'r--', linewidth=2, label=precision_name)
            ax.set_xlim(-2, 2)
            ax.set_ylim(-5, 5)
            ax.set_title(f'{precision_name} PyTorch - Epoch {epoch}, Loss: {loss.item():.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', facecolor='black')
            buf.seek(0)
            frames.append(Image.open(buf).copy())
            buf.close()
    
    plt.close()
    
    # Save GIF
    gif_name = f'poly_{precision_name.lower()}.gif'
    frames[0].save(gif_name, save_all=True, append_images=frames[1:], duration=200, loop=0)
    print(f"{precision_name} PyTorch complete! Final loss: {loss.item():.6f}")
    
    return loss.item()