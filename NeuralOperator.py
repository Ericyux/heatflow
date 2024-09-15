import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from neuralop.models import TFNO
from neuralop import LpLoss

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the data
data = np.load('heat_flow_data_10000_samples.npy')
print(f"Loaded data shape: {data.shape}")

# Prepare the data
# We'll use a subset of the time steps to reduce memory usage
time_step_interval = 2  # Use every 2nd time step
X = data[:, ::time_step_interval, :]  # Subset of timesteps as input
y = data[:, 1::time_step_interval, :]  # Subset of timesteps as target, shifted by one
X = X[:, np.newaxis, :, :]
y = y[:, np.newaxis, :, :]

# Convert to float32 to reduce memory usage
X = X.astype(np.float32)
y = y.astype(np.float32)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.from_numpy(X_train).to(device)
y_train = torch.from_numpy(y_train).to(device)
X_test = torch.from_numpy(X_test).to(device)
y_test = torch.from_numpy(y_test).to(device)

# Create DataLoader with a smaller batch size
batch_size = 8
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define the TFNO model 
model = TFNO(
    n_modes=(16, 16),
    hidden_channels=32, 
    in_channels=1,
    out_channels=1,
    factorization='tucker',
    n_layers=4,  
    domain_padding=(0.1, 0.1) 
).to(device)

# Define loss function and optimizer
criterion = LpLoss(d=3, p=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    # Evaluate on train and test set
    model.eval()
    with torch.no_grad():
        # Use smaller batches for evaluation to avoid OOM
        train_loss = sum(criterion(model(X_train[i:i+batch_size]), y_train[i:i+batch_size]).item() 
                         for i in range(0, len(X_train), batch_size)) / (len(X_train) / batch_size)
        test_loss = sum(criterion(model(X_test[i:i+batch_size]), y_test[i:i+batch_size]).item() 
                        for i in range(0, len(X_test), batch_size)) / (len(X_test) / batch_size)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

# Plot learning curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Learning Curves')
plt.show()

# Evaluate on a random sample
model.eval()
sample_idx = np.random.randint(0, len(X_test))
sample_X = X_test[sample_idx:sample_idx+1]
sample_y = y_test[sample_idx:sample_idx+1]
with torch.no_grad():
    prediction = model(sample_X)

# Plot initial, middle, and final states
plt.figure(figsize=(15, 5))
timesteps = [0, prediction.shape[2]//2, -1]
titles = ['Initial State', 'Middle State', 'Final State']

for i, t in enumerate(timesteps):
    plt.subplot(1, 3, i+1)
    plt.plot(sample_X[0, 0, t, :].cpu().numpy(), label='Input')
    plt.plot(sample_y[0, 0, t, :].cpu().numpy(), label='Actual')
    plt.plot(prediction[0, 0, t, :].cpu().numpy(), label='Predicted')
    plt.title(titles[i])
    plt.xlabel('Position')
    plt.ylabel('Temperature')
    plt.legend()

plt.tight_layout()
plt.show()

# Save the model
torch.save(model.state_dict(), 'heat_flow_neural_operator_model_optimized.pth')
print("Model saved as 'heat_flow_neural_operator_model_optimized.pth'")