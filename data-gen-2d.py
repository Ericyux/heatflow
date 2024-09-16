import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_random_initial_condition_2d(num_points, num_peaks=5, amplitude=1.0):
    """
    Generate a random initial condition with multiple peaks in 2D.
    
    Parameters:
    - num_points: Number of spatial points in each dimension
    - num_peaks: Number of random peaks to generate
    - amplitude: Maximum amplitude of the peaks
    
    Returns:
    - initial_condition: numpy array of shape (num_points, num_points)
    """
    x = np.linspace(0, 1, num_points)
    y = np.linspace(0, 1, num_points)
    X, Y = np.meshgrid(x, y)
    initial_condition = np.zeros((num_points, num_points))
    
    for _ in range(num_peaks):
        peak_x = np.random.rand()
        peak_y = np.random.rand()
        peak_width = 0.1 + 0.2 * np.random.rand()  # Random width between 0.1 and 0.3
        peak = amplitude * np.exp(-((X - peak_x)**2 + (Y - peak_y)**2) / peak_width**2)
        initial_condition += peak
    
    # Normalize to range [0, 1]
    initial_condition = initial_condition / np.max(initial_condition)
    
    return initial_condition

def generate_2d_heat_flow_data(num_points, num_timesteps, diffusivity, dt, dx):
    """
    Generate 2D heat flow data.
    
    Parameters:
    - num_points: Number of spatial points in each dimension
    - num_timesteps: Number of time steps
    - diffusivity: Thermal diffusivity
    - dt: Time step size
    - dx: Spatial step size
    
    Returns:
    - data: numpy array of shape (num_timesteps, num_points, num_points)
    """
    # Initialize temperature array
    T = np.zeros((num_timesteps, num_points, num_points))
    
    # Set random initial condition
    T[0] = generate_random_initial_condition_2d(num_points)
    
    # Set boundary conditions (fixed at 0)
    T[:, 0, :] = T[:, -1, :] = T[:, :, 0] = T[:, :, -1] = 0
    
    alpha = diffusivity
    CFL = alpha * dt / (dx**2)
    # Ensure the CFL condition is met for stability
    print(CFL)
    if CFL > 0.25:
        raise ValueError("The CFL condition is not satisfied. Reduce dt or increase dx.")
    
    # Compute the heat flow
    for t in range(1, num_timesteps):
        T[t, 1:-1, 1:-1] = T[t-1, 1:-1, 1:-1] + CFL * (
            T[t-1, 2:, 1:-1] + T[t-1, :-2, 1:-1] +
            T[t-1, 1:-1, 2:] + T[t-1, 1:-1, :-2] -
            4 * T[t-1, 1:-1, 1:-1]
        )
    
    return T

def generate_multiple_samples(num_samples, num_points, num_timesteps, diffusivity, dt, dx):
    """
    Generate multiple samples of 2D heat flow data.
    
    Parameters:
    - num_samples: Number of samples to generate
    - num_points: Number of spatial points in each dimension
    - num_timesteps: Number of time steps
    - diffusivity: Thermal diffusivity
    - dt: Time step size
    - dx: Spatial step size
    
    Returns:
    - data: numpy array of shape (num_samples, num_timesteps, num_points, num_points)
    """
    data = np.zeros((num_samples, num_timesteps, num_points, num_points))
    
    for i in tqdm(range(num_samples), desc="Generating samples"):
        data[i] = generate_2d_heat_flow_data(num_points, num_timesteps, diffusivity, dt, dx)
    
    return data

# Set parameters
num_samples = 10000
num_points = 100  
num_timesteps = 100  # Reduced from 1000 to 100        
diffusivity = 0.1
dt = 0.0005
dx = 1.5 / (num_points - 1)

# Generate multiple samples
data = generate_multiple_samples(num_samples, num_points, num_timesteps, diffusivity, dt, dx)

# Save the data
np.save('heat_flow_data_2d_10000_samples.npy', data)

print(f"Data shape: {data.shape}")
print("Data saved as 'heat_flow_data_2d_10000_samples.npy'")

# Plot a few random samples
num_plots = 3
fig, axs = plt.subplots(num_plots, 3, figsize=(15, 5*num_plots))
fig.suptitle('Sample 2D Heat Flow Simulations')

for i in range(num_plots):
    sample_idx = np.random.randint(0, num_samples)
    
    # Plot initial condition
    im = axs[i, 0].imshow(data[sample_idx, 0], cmap='hot', extent=[0, 1, 0, 1], vmin=0, vmax=1)
    axs[i, 0].set_title(f'Sample {sample_idx} - Initial Condition')
    axs[i, 0].set_xlabel('X')
    axs[i, 0].set_ylabel('Y')
    plt.colorbar(im, ax=axs[i, 0])
    
    # Plot middle state
    im = axs[i, 1].imshow(data[sample_idx, num_timesteps//2], cmap='hot', extent=[0, 1, 0, 1], vmin=0, vmax=1)
    axs[i, 1].set_title(f'Sample {sample_idx} - Middle State')
    axs[i, 1].set_xlabel('X')
    axs[i, 1].set_ylabel('Y')
    plt.colorbar(im, ax=axs[i, 1])
    
    # Plot final state
    im = axs[i, 2].imshow(data[sample_idx, -1], cmap='hot', extent=[0, 1, 0, 1], vmin=0, vmax=1)
    axs[i, 2].set_title(f'Sample {sample_idx} - Final State')
    axs[i, 2].set_xlabel('X')
    axs[i, 2].set_ylabel('Y')
    plt.colorbar(im, ax=axs[i, 2])

plt.tight_layout()
plt.show()
