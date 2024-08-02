import torch
import torch.nn as nn

# Assuming the device is GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Example labels (ensure dtype is int64 and move to the correct device)
masked_labels = torch.randint(0, 2, (1542, 1), dtype=torch.long).to(device)

print(f"masked_labels shape:{masked_labels.shape}")

# Convert labels to one-hot encoding
num_classes = 2
one_hot_labels = torch.zeros(masked_labels.size(0), num_classes, device=device)  # Create tensor on the correct device

# Flatten masked_labels to match the dimensions for scatter_
masked_labels = masked_labels.squeeze(1)  # Convert from [1542, 1] to [1542]
print(f"masked_labels shape:{masked_labels.shape}")

# Scatter 1s into the correct positions
one_hot_labels.scatter_(1, masked_labels.unsqueeze(1), 1)  # Ensure masked_labels is [1542, 1]

# Verify the shape
print(one_hot_labels.shape)  # Should be [1542, 2]
print(one_hot_labels)

# Define your loss function
loss_BCE = nn.BCEWithLogitsLoss()

# Dummy predictions (ensure predictions are on the correct device)
preds = torch.randn(1542, 2, device=device)  

# Compute loss (ensure one_hot_labels is in float format)
train_loss = loss_BCE(preds, one_hot_labels.float())
