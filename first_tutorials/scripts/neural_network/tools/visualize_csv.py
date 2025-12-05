import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os.path as path

"""
Here we can animate the joint positions from a Top-View, since
we are working with a 2 joint robot which only moves about the x-y plane.
"""

# Use arbitrary 1.0 m arm link lengths
l1, l2 = 1.0, 1.0

# Load CSV data with 'joint1' and 'joint2' data
script_dir = path.dirname(path.abspath(__file__))
dataset_path = path.join(script_dir, "robot_bc_data.csv") 
data = pd.read_csv(dataset_path)  
joint1 = data['joint1'].values
joint2 = data['joint2'].values
num_steps = len(joint1)

# Compute forward kinematics
def forward_kinematics(theta1, theta2):
    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)
    return (0, x1, x2), (0, y1, y2)             # base, joint1, end-effector

# Create figure
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title('2-Joint Planar Arm Animation')

line, = ax.plot([], [], 'k-', lw=3)             # link lines
joints, = ax.plot([], [], 'ro', markersize=8)   # joints

# Update function for animation
def update(frame):
    x, y = forward_kinematics(joint1[frame], joint2[frame])
    line.set_data(x, y)
    joints.set_data(x, y)
    return line, joints

# Create animation
ani = FuncAnimation(fig, update, frames=num_steps, interval=30, blit=True)
plt.show()
