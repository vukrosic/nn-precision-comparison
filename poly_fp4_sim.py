from poly_base import train_and_visualize, PolyNet, target_poly
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import io
import math

print("FP4 PyTorch - Using manual 4-bit float simulation")

def float_to_fp4_simulation(num):
    """Simulates converting a float to a conceptual 4-bit float (1-2-1 format)."""
    if num == 0:
        return 0
    
    # 1. Sign bit (1 bit)
    sign_bit = 0 if num >= 0 else 1
    num = abs(num)
    
    # 2. Exponent and Mantissa (2 bits and 1 bit)
    mantissa, exponent = math.frexp(num)  # mantissa is in [0.5, 1.0)
    
    # Quantize Exponent (2 bits, range -1 to 2)
    exponent_bias = 1
    quantized_exp = max(0, min(3, exponent + exponent_bias))
    
    # Quantize Mantissa (1 bit)
    quantized_mantissa = 1 if (mantissa * 2 - 1) >= 0.5 else 0
    
    # Pack the bits: S E E M
    fp4_val = (sign_bit << 3) | (quantized_exp << 1) | quantized_mantissa
    return fp4_val

def fp4_simulation_to_float(fp4_val):
    """Converts the conceptual 4-bit float back to a Python float."""
    if fp4_val == 0:
        return 0.0
    
    # Unpack bits: S E E M
    sign_bit = (fp4_val >> 3) & 1
    quantized_exp = (fp4_val >> 1) & 3
    quantized_mantissa = fp4_val & 1
    
    # Dequantize exponent
    exponent_bias = 1
    exponent = quantized_exp - exponent_bias
    
    # Dequantize mantissa (add the implicit leading 1)
    mantissa = (1.0 + quantized_mantissa / 2.0) / 2.0
    
    # Reconstruct the float
    num = math.ldexp(mantissa, exponent)
    return -num if sign_bit == 1 else num

def quantize_tensor_fp4(tensor):
    """Apply FP4 simulation to entire tensor"""
    flat = tensor.flatten()
    quantized = torch.zeros_like(flat)
    
    for i, val in enumerate(flat):
        fp4_bits = float_to_fp4_simulation(val.item())
        quantized[i] = fp4_simulation_to_float(fp4_bits)
    
    return quantized.reshape(tensor.shape)

class PolyNetFP4Sim(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, 64, dtype=torch.float32)
        self.fc2 = torch.nn.Linear(64, 64, dtype=torch.float32)
        self.fc3 = torch.nn.Linear(64, 32, dtype=torch.float32)
        self.fc4 = torch.nn.Linear(32, 1, dtype=torch.float32)
        
        # Initialize weights
        torch.nn.init.normal_(self.fc1.weight, 0, 0.1)
        torch.nn.init.normal_(self.fc2.weight, 0, 0.1)
        torch.nn.init.normal_(self.fc3.weight, 0, 0.1)
        torch.nn.init.normal_(self.fc4.weight, 0, 0.1)
    
    def forward(self, x):
        # Quantize weights to FP4 during forward pass
        w1_q = quantize_tensor_fp4(self.fc1.weight)
        x = F.linear(x, w1_q, self.fc1.bias)
        x = F.silu(x)
        
        w2_q = quantize_tensor_fp4(self.fc2.weight)
        x = F.linear(x, w2_q, self.fc2.bias)
        x = F.silu(x)
        
        w3_q = quantize_tensor_fp4(self.fc3.weight)
        x = F.linear(x, w3_q, self.fc3.bias)
        x = F.silu(x)
        
        w4_q = quantize_tensor_fp4(self.fc4.weight)
        x = F.linear(x, w4_q, self.fc4.bias)
        
        return x

def train_and_visualize_fp4_sim(precision_name="FP4-SIM", lr=0.05, epochs=1500):
    """Training with manual FP4 simulation"""
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    x_range = torch.linspace(-2, 2, 200, dtype=torch.float32, device=device).reshape(-1, 1)
    y_true = target_poly(x_range)
    
    model = PolyNetFP4Sim().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    
    frames = []
    save_every = 75
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6, 4), dpi=80)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass with FP4 simulation
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
            ax.set_title(f'{precision_name} - Epoch {epoch}, Loss: {loss.item():.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', facecolor='black')
            buf.seek(0)
            frames.append(Image.open(buf).copy())
            buf.close()
    
    plt.close()
    
    # Save GIF
    gif_name = f'poly_{precision_name.lower().replace("-", "_")}.gif'
    frames[0].save(gif_name, save_all=True, append_images=frames[1:], duration=200, loop=0)
    print(f"{precision_name} complete! Final loss: {loss.item():.6f}")
    
    return loss.item()

if __name__ == "__main__":
    # Test the FP4 simulation first
    test_val = 3.14
    fp4_bits = float_to_fp4_simulation(test_val)
    reconstructed = fp4_simulation_to_float(fp4_bits)
    print(f"Test: {test_val} -> FP4 bits: {fp4_bits:04b} -> {reconstructed:.4f}")
    
    train_and_visualize_fp4_sim()