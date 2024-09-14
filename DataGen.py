import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_random_initial_condition(num_points, num_peaks=5, amplitude=1.0):
    """
    Generate a random initial condition with multiple peaks.
    
    Parameters:
    - num_points: Number of spatial points
    - num_peaks: Number of random peaks to generate
    - amplitude: Maximum amplitude of the peaks
    
    Returns:
    - initial_condition: numpy array of shape (num_points,)
    """
    x = np.linspace(0, 1, num_points)
    initial_condition = np.zeros(num_points)
    
    for _ in range(num_peaks):
        peak_position = np.random.rand()
        peak_width = 0.1 + 0.2 * np.random.rand()  # Random width between 0.1 and 0.3
        peak = amplitude * np.exp(-((x - peak_position) / peak_width) ** 2)
        initial_condition += peak
    
    # Normalize to range [0, 1]
    initial_condition = initial_condition / np.max(initial_condition)

    return initial_condition

def generate_1d_heat_flow_data(num_points, num_timesteps, diffusivity, dt, dx):
    """
    Generate 1D heat flow data.
    
    Parameters:
    - num_points: Number of spatial points
    - num_timesteps: Number of time steps
    - diffusivity: Thermal diffusivity
    - dt: Time step size
    - dx: Spatial step size
    
    Returns:
    - data: numpy array of shape (num_timesteps, num_points)
    """
    # Initialize temperature array
    T = np.zeros((num_timesteps, num_points))
    
    # Set random initial condition
    T[0, :] = generate_random_initial_condition(num_points)
    
    # Set boundary conditions
    T[:, 0] = 0  # Fixed temperature at left boundary
    T[:, -1] = 0  # Fixed temperature at right boundary
    
    alpha = diffusivity
    CFL = alpha * dt / (dx**2)
    # Ensure the CFL condition is met for stability
    if CFL > 0.5:
        raise ValueError("The CFL condition is not satisfied. Reduce dt or increase dx.")
    
    # Compute the heat flow
    # for t in range(1, num_timesteps):
    #     for i in range(1, num_points - 1):
    #         T[t, i] = T[t-1, i] + diffusivity * dt / (dx**2) * \
    #                   (T[t-1, i+1] - 2*T[t-1, i] + T[t-1, i-1])
    for t in range(1, num_timesteps):
        for i in range(1, num_points - 1):
            T[t, i] = T[t-1, i] + CFL * (T[t-1, i+1] - 2 * T[t-1, i] + T[t-1, i-1])
    
    
    return T

def generate_multiple_samples(num_samples, num_points, num_timesteps, diffusivity, dt, dx):
    """
    Generate multiple samples of 1D heat flow data.
    
    Parameters:
    - num_samples: Number of samples to generate
    - num_points: Number of spatial points
    - num_timesteps: Number of time steps
    - diffusivity: Thermal diffusivity
    - dt: Time step size
    - dx: Spatial step size
    
    Returns:
    - data: numpy array of shape (num_samples, num_timesteps, num_points)
    """
    data = np.zeros((num_samples, num_timesteps, num_points))
    
    for i in tqdm(range(num_samples), desc="Generating samples"):
        data[i] = generate_1d_heat_flow_data(num_points, num_timesteps, diffusivity, dt, dx)
    
    return data

# Set parameters
num_samples = 10000
num_points = 100
num_timesteps = 1000         
diffusivity = 0.1
dt = 0.0005
dx = 1.0 / (num_points - 1)

# Generate multiple samples
data = generate_multiple_samples(num_samples, num_points, num_timesteps, diffusivity, dt, dx)

# Save the data
np.save('heat_flow_data_10000_samples.npy', data)

print(f"Data shape: {data.shape}")
print("Data saved as 'heat_flow_data_10000_samples.npy'")

# Plot a few random samples
num_plots = 5
fig, axs = plt.subplots(num_plots, 2, figsize=(15, 5*num_plots))
fig.suptitle('Sample Heat Flow Simulations')        

for i in range(num_plots):
    sample_idx = np.random.randint(0, num_samples)
    
    # Plot initial condition
    axs[i, 0].plot(np.linspace(0, 1, num_points), data[sample_idx, 0, :])
    axs[i, 0].set_title(f'Sample {sample_idx} - Initial Condition')
    axs[i, 0].set_xlabel('Position')
    axs[i, 0].set_ylabel('Temperature')
    
    # Plot heat flow
    im = axs[i, 1].imshow(data[sample_idx].T, aspect='auto', cmap='hot', extent=[0, num_timesteps*dt, 0, 1], vmin=0, vmax=1)
    axs[i, 1].set_title(f'Sample {sample_idx} - Heat Flow')
    axs[i, 1].set_xlabel('Time')
    axs[i, 1].set_ylabel('Position')
    plt.colorbar(im, ax=axs[i, 1], label='Temperature')

plt.tight_layout()
plt.show()
