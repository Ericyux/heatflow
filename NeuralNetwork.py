import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from neuralop import LpLoss

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the data
data = np.load('heat_flow_data_10000_samples.npy')
print(f"Loaded data shape: {data.shape}")

# Prepare the data
X = data[:, :-1, :]  # All timesteps except the last as input
y = data[:, 1:, :]   # All timesteps except the first as target
X = X[:, np.newaxis, :, :]
y = y[:, np.newaxis, :, :]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train).to(device)
y_train = torch.FloatTensor(y_train).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_test = torch.FloatTensor(y_test).to(device)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Define the modified neural network
class ModifiedHeatFlowNN(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels):
        super(ModifiedHeatFlowNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(hidden_channels, output_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return x

# Initialize the model, loss function, and optimizer
input_channels = 1
output_channels = 1
hidden_channels = 32
model = ModifiedHeatFlowNN(input_channels, output_channels, hidden_channels).to(device)

# Use Xavier initialization for weights
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

model.apply(init_weights)

criterion = LpLoss(d=3, p=2)  # Using 3D loss for time-space data
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
        train_loss = criterion(model(X_train), y_train).item()
        test_loss = criterion(model(X_test), y_test).item()
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    
    if (epoch + 1) % 10 == 0:
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
prediction = model(sample_X)

# Plot initial, middle, and final states
plt.figure(figsize=(15, 5))
timesteps = [0, prediction.shape[2]//2, -1]
titles = ['Initial State', 'Middle State', 'Final State']

for i, t in enumerate(timesteps):
    plt.subplot(1, 3, i+1)
    plt.plot(sample_X[0, 0, t, :].cpu().numpy(), label='Input')
    plt.plot(sample_y[0, 0, t, :].cpu().numpy(), label='Actual')
    plt.plot(prediction[0, 0, t, :].detach().cpu().numpy(), label='Predicted')
    plt.title(titles[i])
    plt.xlabel('Position')
    plt.ylabel('Temperature')
    plt.legend()

plt.tight_layout()
plt.show()

# Save the model
torch.save(model.state_dict(), 'heat_flow_nn_model.pth')
print("Model saved as 'heat_flow_nn_model.pth'")
