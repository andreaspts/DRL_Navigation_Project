'''
Created on 19.04.2019

@author: Andreas
'''

# We import the relevant packages:
import torch
import torch.nn as nn
import torch.nn.functional as F


# We define the deep neural network as a class:
class QNetwork(nn.Module):
    """Actor (Policy) Model via a deep neural network."""

    def __init__(self, state_size, action_size, seed):
        """We initialize the parameters and build the neural network model.
        Arguments:
        ======
            state_size (int): This sets the dimension of each state.
            action_size (int): This is the dimension of each action.
            seed (int): This sets the random seed.
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # We define the layers of the network, 37 states as input, 5 layers, ending with a number of action_size outputs.
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        # Output layer, number of units corresponds to action_size, hence one for each action of q(state_fixed, action).
        self.fc5 = nn.Linear(8, action_size)

    def forward(self, state):
        """Building a network (forward pass) that maps state -> action values."""
        
        state = self.fc1(state)
        state = F.relu(state)
        state = self.fc2(state)
        state = F.relu(state)
        state = self.fc3(state)
        state = F.relu(state)
        state = self.fc4(state)
        state = F.relu(state)
        state = self.fc5(state)
        
        return state
