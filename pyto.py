from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch import tensor
from torch import optim

import matplotlib.pyplot as plt
from mlp import dataset

torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# XOR gate inputs and outputs.
x, y = dataset[0] , dataset[1]
X = xor_input = tensor(x).float().to(device)
Y = xor_output = tensor(y).float().to(device)


# Use tensor.shape to get the shape of the matrix/tensor.
num_data, input_dim = X.shape
print('Inputs Dim:', input_dim) # i.e. n=2 

num_data, output_dim = Y.shape
print('Output Dim:', output_dim) 
print('No. of Data:', num_data) # i.e. n=4

# Step 1: Initialization. 

# Initialize the model.
# Set the hidden dimension size.
hidden_dim = 5
# Use Sequential to define a simple feed-forward network.
model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),

            #nn.Linear(hidden_dim, hidden_dim),
            #nn.Sigmoid(),
            
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
model

# Initialize the optimizer
learning_rate = 0.3
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Initialize the loss function.
criterion = nn.MSELoss()

# Initialize the stopping criteria
# For simplicity, just stop training after certain no. of epochs.
num_epochs = 5000 

losses = [] # Keeps track of the loses.

# Step 2-4 of training routine.

for _e in tqdm(range(num_epochs)):
    # Reset the gradient after every epoch. 
    optimizer.zero_grad() 
    # Step 2: Foward Propagation
    predictions = model(X)

    # Step 3: Back Propagation 
    # Calculate the cost between the predictions and the truth.
    loss = criterion(predictions, Y)
    # Remember to back propagate the loss you've computed above.
    loss.backward()

    # Step 4: Optimizer take a step and update the weights.
    optimizer.step()

    # Log the loss value as we proceed through the epochs.
    losses.append(loss.data.item())
    print(loss.data.item())


plt.plot(losses)
