import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        #advantage function
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, action_size)
        #value function
        self.fc_v1 = nn.Linear(state_size,32)
        self.fc_v2 = nn.Linear(32,1)
    def forward(self, state):
        """Build a network that maps state -> action values."""
        advantage = F.relu(self.fc1(state))
        advantage = self.fc2(advantage)

        Value = F.relu(self.fc_v1(state))
        Value = self.fc_v2(Value)

        return Value + (advantage - torch.mean(advantage,1,keepdim=True))
