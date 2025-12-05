import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from model_cfg import SimpleMLP
import os.path as path
import matplotlib.pyplot as plt

"""
Basic training script using SimpleMLP (model.py)

1. State-Action pair data is loaded from CSV with columns: "joint1", "joint2", "d_joint1", "d_joint2".
2. This creates input tensor X with shape [N, 2] and target tensor Y with shape [N, 2]. (N Samples)
3. PyTorch wraps (X, Y) in a TensorDataset and use DataLoader to produce minibatches.
5. The model is chosen, and the loss function, and optimizer are defined.
6. The model is trained for num_epochs: each epoch iterates over the dataset once (in minibatches), with the goal to minimize the loss.
7. At each minibatch: forward pass --> compute loss --> backward() --> optimizer.step().
8. After the number of epochs, the training stops, and the loss value is used as a metric to evaluate the accuracy of the trained model (lower loss -> lower error -> better)

The trained model is saved as "simple_mlp.pth", where the .pth format saves the final (weights, biases, etc.).

Notes
- This is a very simple model, SO many things are not considered...
- Normalize inputs using training-set mean/std for stable training.
- Split dataset into train/validation/test to monitor generalization.
- Tune hyperparameters: hidden_size, batch_size, lr, optimizer, weight_decay, epochs.
- Typical monitoring: training and validation loss per epoch; inspect predictions qualitatively.
"""

# Set path for trained model
script_dir = path.dirname(path.abspath(__file__))
trained_model_path = path.join(script_dir, "trained_models","simple_mlp.pth")

# Load CSV dataset with joint1, joint2, d_joint1, d_joint2
dataset_path = path.join(script_dir, "tools","robot_bc_data.csv")
data = pd.read_csv(dataset_path)

inputs = torch.tensor(data[['joint1','joint2']].values, dtype=torch.float32)        # X
outputs = torch.tensor(data[['d_joint1','d_joint2']].values, dtype=torch.float32)   # Y

dataset = TensorDataset(inputs, outputs)
loader = DataLoader(dataset, batch_size=16, shuffle=True)           # DataLoader creates (x_batch, y_batch) with shapes [batch_size,2]

# Create model for training (parameters here can be varied, also referred to as "hyper-parameters")
model = SimpleMLP(input_size=2, hidden_size=16, output_size=2)      # here the weights and biases are initialized by default
criterion = torch.nn.MSELoss()                                      # MSE: Mean-Squared Loss criterion is used (to check prediction and y_batch)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)           # We use a Learning Rate (LR) of 0.01, to decide by how much to control the step size per optimizer update

# Training loop which will run for num_epochs
num_epochs = 100                                                    # Each epoch, means a whole sweep from start to end of the MLP (Outer loop)
train_losses = []
for epoch in range(num_epochs):                                     # Within each epoch, an inner loop runs
    running = 0.0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()                                       # clears accumulated gradients
        pred = model(x_batch)                                       # forward pass to obtain a prediction [batch_size, 2]
        loss = criterion(pred, y_batch)                             # A scalar tensor to calculate the error between the prediction and expected y_batch value
        loss.backward()                                             # Computes the gradients d_loss/d_param
        optimizer.step()                                            # Updates the parameters using gradients and Adam rule
        running+=loss.item()*x_batch.size(0)
    train_loss = running/len(loader.dataset)
    train_losses.append(train_loss)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# Save a plot of the training progress
plt.plot(train_losses, label="train")
plt.xlabel("epoch"); plt.ylabel("MSE loss"); plt.legend()
plt.savefig(path.join(script_dir,"loss_chart.png"), dpi=150)

# Save the model in a portable format, such that it can easily be called for inference.
torch.save(model.state_dict(), trained_model_path)