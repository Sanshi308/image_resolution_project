import torch
import torch.nn as nn
from utils import SimpleSRCNN

# Initialize model
model = SimpleSRCNN(scale_factor=2)

# Dummy loss & optimizer (ya aap real training loop chala sakte ho)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Save trained weights
torch.save(model.state_dict(), "sr_model.pth")
print("Model weights saved as sr_model.pth!")