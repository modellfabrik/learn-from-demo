import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from model_cfg import SimpleMLP
import os.path as path

"""
Basic Inference Script using Trained Model

Using a trained simple MLP model, we predict the next joint positions of 
the planar 2-joint robot arm based on current joint angles.

We use a 2D (x-y) plot, with 2 Links seen from Top-View.

1. First read the CSV file containing previous joint data to reuse input values and ranges.
2. We extract the relevant joint columns ('joint1', 'joint2') from the dataset and convert them to a PyTorch tensor for model input.
3. Load the model architecture through the SimpleMLP network (with input, hidden, and output sizes.) We will use the model to evaluate/predict.
4. We select one input sample from the time-series (specific row from the inputs as the current joint angles.)
5. The prediction then runs to estimate the next increment (in `torch.no_grad()` context to avoid computing gradients)
6. We convert predictions to NumPy arrays by flattening the current and predicted joint angles for further processing.
7. Define forward kinematics (Compute 2D position with link lengths L1,L2=1,1)
8. We plot top-down view to visualize the current and predicted arm position.
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

# Select one input row (e.g., index 0)
current_input = inputs[0].unsqueeze(0)              # [1, 2] --> Adds a batch dimension using `unsqueeze`

# Predict next increment using the model
with torch.no_grad():
    next_increment = model(current_input)           # this is specifically where inference happens (Use model to predict).
    next_joints = current_input + next_increment    # add the predicted increment to the current joint angles.

# Convert to numpy
curr_angles = current_input.numpy().flatten()
next_angles = next_joints.numpy().flatten()

# Forward kinematics function for planar 2-joint arm
def forward_kinematics(theta1, theta2, l1=1.0, l2=1.0):
    x0, y0 = 0, 0
    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)
    return np.array([[x0, y0], [x1, y1], [x2, y2]])

# Compute joint positions
curr_pos = forward_kinematics(curr_angles[0], curr_angles[1])
next_pos = forward_kinematics(next_angles[0], next_angles[1])

print("Current joint angles:", current_input.numpy())
print("Predicted increments:", next_increment.numpy())
print("Predicted next joint angles:", next_joints.numpy())

# Plot top-down view
plt.figure(figsize=(6,6))
plt.plot(curr_pos[:,0], curr_pos[:,1], 'k-', label='Current')
plt.plot(next_pos[:,0], next_pos[:,1], 'r-', label='Predicted Next')
plt.scatter(curr_pos[:,0], curr_pos[:,1], color='red')
plt.scatter(next_pos[:,0], next_pos[:,1], color='blue')
plt.xlim(-2.2, 2.2)
plt.ylim(-2.2, 2.2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Planar 2-Joint Arm: Current (red) vs Predicted Next (blue)')
plt.legend()
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
