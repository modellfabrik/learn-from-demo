import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from model_cfg import SimpleMLP
import os.path as path

"""
Closed-Loop Inference Script

What we did so far is used the synthetic data which is perfect for prediction.
What happens if we use the predicted positions as feedback?

        current pos --> model --> predict new pos --> model --> predict new pos --> ...
"""

# Load dataset (just to reuse the input values and range)
script_dir = path.dirname(path.abspath(__file__))
data_path = path.join(script_dir,"tools","robot_bc_data.csv")
data = pd.read_csv(data_path)
expert_joints = data[['joint1','joint2']].values

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

num_steps = 1000
predicted_joints = []

current_input = torch.tensor(expert_joints[0], dtype=torch.float32).unsqueeze(0)

# Closed-loop BC-type prediction
with torch.no_grad():
    for _ in range(num_steps):
        increment = model(current_input)
        next_joint = current_input + increment
        predicted_joints.append(next_joint.numpy().flatten())
        current_input = next_joint

predicted_joints = np.array(predicted_joints)

## Animation
fig, ax = plt.subplots(figsize=(6,6))
pred_line, = ax.plot([], [], 'b-', lw=2, label='BC Prediction')
pred_points, = ax.plot([], [], 'bo', markersize=6)
expert_line, = ax.plot([], [], 'r-', lw=2, label='Expert Trajectory')
expert_points, = ax.plot([], [], 'ro', markersize=6)

ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Behavior Cloning vs Expert Trajectory')
ax.grid(True)
ax.set_aspect('equal')
ax.legend()

def init():
    pred_line.set_data([], [])
    pred_points.set_data([], [])
    expert_line.set_data([], [])
    expert_points.set_data([], [])
    return pred_line, pred_points, expert_line, expert_points

def update(frame):
    # BC predicted positions
    theta1, theta2 = predicted_joints[frame]
    pred_pos = forward_kinematics(theta1, theta2)
    pred_line.set_data(pred_pos[:,0], pred_pos[:,1])
    pred_points.set_data(pred_pos[:,0], pred_pos[:,1])
    
    # Expert positions
    theta1_e, theta2_e = expert_joints[frame]
    expert_pos = forward_kinematics(theta1_e, theta2_e)
    expert_line.set_data(expert_pos[:,0], expert_pos[:,1])
    expert_points.set_data(expert_pos[:,0], expert_pos[:,1])
    
    return pred_line, pred_points, expert_line, expert_points

ani = FuncAnimation(fig, update, frames=num_steps, init_func=init,
                    blit=True, interval=100, repeat=False)

plt.show()