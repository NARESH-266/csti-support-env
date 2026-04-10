import numpy as np
from typing import Dict, Any, List, Tuple

class TrainingSimulator:
    """Simulates the 'physics' of an ML training run without actual computation."""
    
    def __init__(self, hidden_bug: str):
        self.hidden_bug = hidden_bug
        self.epoch = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "grad_norm": []
        }

    def simulate(self, config: Dict[str, Any], steps: int) -> Tuple[Dict[str, List[float]], List[str]]:
        """Advance the training simulation by N steps."""
        logs = []
        
        lr = config.get("learning_rate", 1e-3)
        use_bn = config.get("use_batch_norm", False)
        depth = config.get("num_layers", 5)
        dropout = config.get("dropout", 0.0)
        
        # 🟢 Base convergence properties
        # Good LR is usually between 1e-4 and 1e-2
        lr_factor = 1.0
        if lr > 0.05:
            lr_factor = -1.0 # Divergence
            logs.append(f"Epoch {self.epoch + 1}: [WARNING] Loss is increasing! NaN detected in gradients.")
        elif lr < 1e-6:
            lr_factor = 0.01 # Extremely slow
            logs.append(f"Epoch {self.epoch + 1}: [INFO] Training is very slow. Learning rate might be too low.")
        
        # 🟢 Vanishing Gradient Logic (Task 2)
        grad_factor = 1.0
        if depth > 15 and not use_bn and self.hidden_bug == "vanishing_gradient":
            grad_factor = 0.05
            logs.append(f"Epoch {self.epoch + 1}: [WARNING] Gradients for early layers are near zero.")
        
        # 🟢 Simulation loop
        for _ in range(steps):
            self.epoch += 1
            
            # Progress follows a logarithmic curve if convergence is healthy
            progress = (1 - np.exp(-0.1 * self.epoch * lr_factor * grad_factor))
            
            # Base accuracy capped at 0.95
            base_acc = 0.95 * progress if lr_factor > 0 else 0.1
            
            # 🔴 Hidden Bug: Data Leakage (Task 3)
            # If bug is leakage, Val Acc is suspiciously perfect but Train Loss is high
            if self.hidden_bug == "data_leakage":
                val_acc = 0.999
                train_loss = 2.3 * (1 - progress) + 0.1
                logs.append(f"Epoch {self.epoch}: [INFO] Validation accuracy is unusually high: 99.9%")
            else:
                val_acc = base_acc
                train_loss = 2.3 * (1 - progress) + 0.1
            
            self.history["train_loss"].append(float(train_loss))
            self.history["val_accuracy"].append(float(val_acc))
            self.history["grad_norm"].append(float(0.5 * grad_factor + np.random.normal(0, 0.05)))
        
        return self.history, logs
