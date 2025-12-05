import torch.nn as nn

"""
Simple Multi-Layer Perceptron (MLP) implemented in PyTorch.
Note: Also referred to as Neural Network or "AI-architecture"...

Architecture:
- Input Layer:      2 neurons (joint1, joint2)
- Hidden Layer 1:   16 neurons, ReLU activation
- Hidden Layer 2:   16 neurons, ReLU activation
- Output Layer:     2 neurons (predicted joint increments)

Notes:
- Hidden layer size (number of neurons, here 16) can be adjusted based on task complexity.
- Too few neurons may cause underfitting (model cannot learn complex task well).
- Too many neurons for a simple task may cause overfitting (model memorizes data instead of generalizing).
"""

class SimpleMLP(nn.Module):
    def __init__(self, input_size=2, hidden_size=16, output_size=2):
        super(SimpleMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),     # fully connected layer mapping input features (2) --> hidden neurons (16)
            nn.ReLU(),                              # applies non-linear ReLu (Rectified Linear) activation for complex mapping. ReLu: f(x) = max(0,x)
            nn.Linear(hidden_size, hidden_size),    # fully connected second layer: 16 neurons --> 16 neurons
            nn.ReLU(),                              # apply ReLu function again
            nn.Linear(hidden_size, output_size)     # final layer mapping hidden layer (16) --> output (2)
        )
    
    def forward(self, x):                           # takes input x, and passes it through each layer sequentially
        return self.model(x)
    

"""
Notes on how ReLu Activation Function is working:
f(x) = max(0,x)
- If input x is positive, it passes unchanged
- If input x is negative, it outputs 0

Other activation functions: Sigmoid, Tanh, Leaky ReLu, Softmax
"""