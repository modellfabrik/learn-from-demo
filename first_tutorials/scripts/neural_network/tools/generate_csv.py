import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as path

"""
This script generates a pseudo-dataset, that could be used for training an Imitation Learning (IL) model.
It creates a state --> action set, whereby:
states:     joint positions at step t
actions:    incremental changes in joint position at step t + 1

policy will do this:
* policy(state) --> action
* next joint position = current joint position + (action * scale factor)

A scale factor is used to change the magnitude of the step.
"""

script_dir = path.dirname(path.abspath(__file__))
dataset_path = path.join(script_dir, "robot_bc_data.csv") 

# Parameters
num_samples = 1000
t = np.linspace(0, 10, num_samples)     # time vector (0 to 10 seconds)
dt = t[1] - t[0]                        # time step size (10 / 1000 ~ 0.01 s)

# states: Generate joint data using a repetitive behaviour
joint1 = np.sin(t)
joint2 = np.cos(t)
scale_factor = 5.0

# actions: compute increments (next value - current value)
delta_joint1 = np.diff(joint1, prepend=joint1[0]) * scale_factor
delta_joint2 = np.diff(joint2, prepend=joint2[0]) * scale_factor

# Combine into a Pandas DataFrame
df = pd.DataFrame({
    'joint1': joint1,
    'joint2': joint2,
    'd_joint1': delta_joint1,
    'd_joint2': delta_joint2
})

# Save dataset as CSV
csv_filename = dataset_path
df.to_csv(csv_filename, index=False)
print(f"CSV file '{csv_filename}' generated successfully at {csv_filename}!")

# Display joint values in graphical form
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(t, joint1, label='joint1 (sin)')
plt.plot(t, joint2, label='joint2 (cos)')
plt.xlabel('Time')
plt.ylabel('Joint angles')
plt.title('Joint Inputs')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(t, delta_joint1, label='d_joint1')
plt.plot(t, delta_joint2, label='d_joint2')
plt.xlabel('Time')
plt.ylabel('Joint increments')
plt.title('Output Increments')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
