import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from neuralop import LpLoss
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the data
data = np.load('heat_flow_data_2d_10000_samples.npy')
print(f"Loaded data shape: {data.shape}")

# Prepare the data
time_step_interval = 5
X = data[:, ::time_step_interval, :, :]
y = data[:, 1::time_step_interval, :, :]
X = X[:, np.newaxis, :, :, :]
y = y[:, np.newaxis, :, :, :]

X = X.astype(np.float32)
y = y.astype(np.float32)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.from_numpy(X_train).to(device)
y_train = torch.from_numpy(y_train).to(device)
X_test = torch.from_numpy(X_test).to(device)
y_test = torch.from_numpy(y_test).to(device)

# Create DataLoader
batch_size = 8
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

class ImprovedHeatFlowNN2D(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels):
        super(ImprovedHeatFlowNN2D, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv6 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv7 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv8 = nn.Conv3d(hidden_channels, output_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(hidden_channels)
        self.bn2 = nn.BatchNorm3d(hidden_channels)
        self.bn3 = nn.BatchNorm3d(hidden_channels)
        self.bn4 = nn.BatchNorm3d(hidden_channels)
        self.bn5 = nn.BatchNorm3d(hidden_channels)
        self.bn6 = nn.BatchNorm3d(hidden_channels)
        self.bn7 = nn.BatchNorm3d(hidden_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.conv8(x)
        return x

input_channels = 1
output_channels = 1
hidden_channels = 80
model = ImprovedHeatFlowNN2D(input_channels, output_channels, hidden_channels).to(device)

criterion = LpLoss(d=4, p=2)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)

num_epochs = 50
train_losses = []
test_losses = []

# Create a directory for checkpoints
os.makedirs('checkpoints', exist_ok=True)

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
    
    model.eval()
    with torch.no_grad():
        train_loss = sum(criterion(model(X_train[i:i+batch_size]), y_train[i:i+batch_size]).item() 
                         for i in range(0, len(X_train), batch_size)) / (len(X_train) / batch_size)
        test_loss = sum(criterion(model(X_test[i:i+batch_size]), y_test[i:i+batch_size]).item() 
                        for i in range(0, len(X_test), batch_size)) / (len(X_test) / batch_size)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    
    scheduler.step(test_loss)
    
    # Display loss for every epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
    
    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        checkpoint_path = f'checkpoints/neural_network_checkpoint_epoch_{epoch+1}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Learning Curves - Neural Network')
plt.show()

torch.save(model.state_dict(), 'heat_flow_nn_model_2d.pth')
print("Model saved as 'heat_flow_nn_model_2d.pth'")

print(f"Final Test Loss (Neural Network): {test_losses[-1]:.4f}")