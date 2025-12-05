import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from model_cfg import SimpleMLP
import os.path as path

"""
Animated Inference Script

Instead of predicting one instance, we can predict multiple and display it using an animated 2D Plot
"""

# Load dataset (just to reuse the input values and range)
script_dir = path.dirname(path.abspath(__file__))
data_path = path.join(script_dir,"tools","robot_bc_data.csv")
data = pd.read_csv(data_path)
inputs = torch.tensor(data[['joint1','joint2']].values, dtype=torch.float32)

# Load our trained model
trained_model_path = path.join(script_dir,"trained_models","simple_mlp.pth")
model = SimpleMLP(input_size=2, hidden_size=16, output_size=2)
model.load_state_dict(torch.load(trained_model_path,  weights_only=True))
model.eval()

def forward_kinematics(theta1, theta2, l1=1.0, l2=1.0):
    x0, y0 = 0, 0
    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)
    return np.array([[x0, y0], [x1, y1], [x2, y2]])

fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title('2-Joint Planar Arm: Current vs Predicted Next')

line_curr, = ax.plot([], [], 'k-', lw=3, label='Current')
line_next, = ax.plot([], [], 'r-', lw=3, label='Predicted Next')
joints_curr, = ax.plot([], [], 'ro', markersize=8)
joints_next, = ax.plot([], [], 'bo', markersize=8)

ax.legend()

# Update function for animation
def update(frame):
    # Current joint angles
    current_input = inputs[frame].unsqueeze(0)
    # Predict next increment
    with torch.no_grad():
        next_increment = model(current_input)
        next_joints = current_input + next_increment
    curr_pos = forward_kinematics(current_input[0,0].item(), current_input[0,1].item())
    next_pos = forward_kinematics(next_joints[0,0].item(), next_joints[0,1].item())
    
    line_curr.set_data(curr_pos[:,0], curr_pos[:,1])
    line_next.set_data(next_pos[:,0], next_pos[:,1])
    joints_curr.set_data(curr_pos[:,0], curr_pos[:,1])
    joints_next.set_data(next_pos[:,0], next_pos[:,1])
    
    return line_curr, line_next, joints_curr, joints_next

#num_frames = min(100, len(inputs)) # animate over specific range
num_frames = len(inputs)            # animate over entire dataset
ani = FuncAnimation(fig, update, frames=num_frames, interval=200, blit=True)

plt.show()