import numpy as np
import torch
import matplotlib.pyplot as plt
from data_gen_2d import generate_2d_heat_flow_data
from neural_network_2d import HeatFlowNN2D
from neuralop.models import TFNO

# Function to calculate Mean Squared Error (MSE)
def calculate_mse(ground_truth, prediction):
    return np.mean((ground_truth - prediction) ** 2)

def load_checkpoint(filepath, model):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'), weights_only=True)
    state_dict = checkpoint['model_state_dict']
    
    # Remove the 'module.' prefix if it exists
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict)
    return model

def generate_sample():
    num_points = 100
    num_timesteps = 1000
    diffusivity = 0.1
    dt = 0.0005
    dx = 1.5 / (num_points - 1)
    return generate_2d_heat_flow_data(num_points, num_timesteps, diffusivity, dt, dx)

def prepare_input(sample):
    time_step_interval = 5
    X = sample[::time_step_interval]  # Reduce timesteps to match model input requirements
    X = X[np.newaxis, np.newaxis, :, :, :]  # Add batch and channel dimensions
    return torch.from_numpy(X.astype(np.float32))

def prepare_ground_truth(sample, time_step_interval=5):
    # Reduce the timesteps in the ground truth to match the predictions
    return sample[::time_step_interval]

def get_prediction(model, X):
    with torch.no_grad():
        return model(X).squeeze().numpy()

def visualize_comparison(ground_truth, nn_predictions, no_predictions, checkpoints, nn_mse, no_mse):
    num_checkpoints = len(checkpoints)
    fig, axs = plt.subplots(3, num_checkpoints, figsize=(20, 12))
    fig.suptitle('Comparison of Ground Truth, Neural Network, and Neural Operator')

    vmin, vmax = 0, 1  # Assuming the data is normalized between 0 and 1

    for i, checkpoint in enumerate(checkpoints):
        im = axs[0, i].imshow(ground_truth[-1], aspect='equal', cmap='hot', vmin=vmin, vmax=vmax)
        axs[0, i].set_title(f'Ground Truth\nFinal State')
        axs[0, i].set_xlabel('X')
        axs[0, i].set_ylabel('Y')
        plt.colorbar(im, ax=axs[0, i])

        im = axs[1, i].imshow(nn_predictions[i][-1], aspect='equal', cmap='hot', vmin=vmin, vmax=vmax)
        axs[1, i].set_title(f'Neural Network\nCheckpoint {checkpoint}\nMSE: {nn_mse[i]:.6f}')
        axs[1, i].set_xlabel('X')
        axs[1, i].set_ylabel('Y')
        plt.colorbar(im, ax=axs[1, i])

        im = axs[2, i].imshow(no_predictions[i][-1], aspect='equal', cmap='hot', vmin=vmin, vmax=vmax)
        axs[2, i].set_title(f'Neural Operator\nCheckpoint {checkpoint}\nMSE: {no_mse[i]:.6f}')
        axs[2, i].set_xlabel('X')
        axs[2, i].set_ylabel('Y')
        plt.colorbar(im, ax=axs[2, i])

    plt.tight_layout()
    plt.show()

# Generate a sample
sample = generate_sample()
ground_truth = prepare_ground_truth(sample)  # Prepare the ground truth with the same time intervals
X = prepare_input(sample)

# Load models
nn_model = HeatFlowNN2D(1, 1, 80)
no_model = TFNO(n_modes=(12, 12, 12), hidden_channels=32, in_channels=1, out_channels=1,
                factorization='tucker', n_layers=4, domain_padding=(0.1, 0.1, 0.1))

checkpoints = [10, 20, 30, 40, 50]
nn_predictions = []
no_predictions = []
nn_mse = []  # Store MSE values for the neural network
no_mse = []  # Store MSE values for the neural operator

for checkpoint in checkpoints:
    nn_checkpoint_path = f'checkpoints/neural_network_checkpoint_epoch_{checkpoint}.pth'
    no_checkpoint_path = f'checkpoints/neural_operator_checkpoint_epoch_{checkpoint}.pth'

    nn_model = load_checkpoint(nn_checkpoint_path, nn_model)
    no_model = load_checkpoint(no_checkpoint_path, no_model)

    nn_pred = get_prediction(nn_model, X)
    no_pred = get_prediction(no_model, X)

    nn_predictions.append(nn_pred)
    no_predictions.append(no_pred)

    # Calculate MSE for both models
    nn_mse.append(calculate_mse(ground_truth, nn_pred))
    no_mse.append(calculate_mse(ground_truth, no_pred))

    # Print MSE for each epoch
    print(f"Epoch {checkpoint}:")
    print(f"  Neural Network MSE: {nn_mse[-1]:.6f}")
    print(f"  Neural Operator MSE: {no_mse[-1]:.6f}")

# Visualize the comparison with MSE
visualize_comparison(ground_truth, nn_predictions, no_predictions, checkpoints, nn_mse, no_mse)
