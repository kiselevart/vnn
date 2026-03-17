import torch
import torch.nn as nn
import torch.optim as optim
import math
from utils.model_factory import get_model

class Args:
    task = "video"
    model = "vnn_fusion"
    dataset = "ucf101"
    num_classes = 101
    device = "cpu"
    lr = 0.001 # Reduced LR for stability
    weight_decay = 1e-4

def test_vnn_fusion_stability():
    args = Args()
    device = torch.device("cpu")
    model = get_model(args, device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Synthetic inputs: (rgb, flow)
    # rgb: (B, 3, 16, 112, 112)
    # flow: (B, 2, 16, 112, 112)
    B = 2
    # Use smaller standard deviation for inputs to prevent initial saturation
    rgb = torch.randn(B, 3, 16, 112, 112) * 0.1
    flow = torch.randn(B, 2, 16, 112, 112) * 0.1
    targets = torch.randint(0, 101, (B,))
    
    print(f"Starting stability test for {args.model}...")
    
    prev_loss = float('inf')
    
    for i in range(20):
        optimizer.zero_grad()
        
        outputs = model((rgb, flow))
        loss = criterion(outputs, targets)
        
        # Diagnostics
        out_max = outputs.abs().max().item()
        is_finite = torch.isfinite(outputs).all().item()
        
        if not is_finite:
            print(f"Step {i}: Model produced non-finite outputs!")
            return False
            
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Check gradients
        grad_norms = []
        for p in model.parameters():
            if p.grad is not None:
                grad_norms.append(p.grad.norm().item())
        
        avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
        
        print(f"Step {i}: Loss={loss.item():.4f}, Max|out|={out_max:.2e}, AvgGradNorm={avg_grad_norm:.2e}")
        
        if math.isnan(loss.item()):
            print(f"Step {i}: Loss is NaN!")
            return False
            
        optimizer.step()
        
    print("Stability test completed.")
    return True

if __name__ == "__main__":
    test_vnn_fusion_stability()
