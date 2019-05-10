'''
Created on 19.04.2019

@author: Andreas
'''


# We import the relevant packages. In particular, we require the QNetwork class from the model.py file:
import numpy as np
import random
from collections import namedtuple, deque
from model import QNetwork
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# We set the basic parameters of the agent:
# The replay buffer size is:
BUFFER_SIZE = int(1e6)
# The minibatch size is:
BATCH_SIZE = 64         
# The discount factor is:
GAMMA = 0.99       
# The parameter for the soft update of target parameters is set to:     
TAU = 1e-3      
# The learning rate is:        
LR = 5e-4
# The gradient descent update rate of the network is:
UPDATE_EVERY = 4


# We set the device either to GPU or CPU:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# We define the agent class:
class Agent():
    """The Agent class encodes how the agent interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, seed):
        """We fist initialize an Agent object.
        
        Arguments:
        =========
            state_size (of int type): This defines the dimension of each state.
            action_size (of int type): This defines the dimension of each action.
            seed (of int type): This sets the random seed.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Setting up the local and target Q-Networks and assigning the computation device:
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        # Setting up the backward propagation on the local network:
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # The definition of the replay memory, as retrieved from the ReplayBuffer class following below.
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # We initialize the time step (for updating every UPDATE_EVERY steps):
        self.t_step = 0
        
        
    def step(self, state, action, reward, next_state, done):
        # We save the experience in the replay memory.
        self.memory.add(state, action, reward, next_state, done)
        
        # The agent learna every UPDATE_EVERY time steps:
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # In case enough samples are available in the memory, we get a random subset and learn from it:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
         
                
    def act(self, state, eps=0.):
        """This method returns actions for a given state as per current policy.
        
        Arguments:
        =========
            state (which is array_like): This is the current state.
            eps (of type float): This corresponds to epsilon, needed in epsilon-greedy action selection.
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # The Epsilon-greedy action selection is done via:
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
 
        Arguments:
        =========
            experiences (Tuple[torch.Variable]): This is a tuple of (s, a, r, s', done) tuples.
            gamma (float): This is the discount factor.
        """
        states, actions, rewards, next_states, dones = experiences
 
        # Forward and backward passes fed with the states and actions. We use an MSELoss loss function and an Adam optimizer:
        output = self.qnetwork_local.forward(states).gather(1,actions)
        self.criterion = nn.MSELoss()
        # The loss function is fed with values from the local network and the target network (as defined in a method below):
        loss = self.criterion(output, self.targets(gamma, rewards,next_states,dones))
         
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
             
        # ------------------- Finally, we have to update the target network. ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     
 
 
    def targets(self,gamma,rewards,next_states,dones):
        
        # We use our target network in order to calculate the target Q-value of taking an action at the next state.
        with torch.no_grad():
            q = self.qnetwork_target.forward(next_states)
        
        # The TD target is calculated and then returned (the factor  (1 - dones) is crucial):
        y = rewards + torch.mul(torch.max(q,dim=1,keepdim=True)[0],gamma) * (1 - dones)
        
        return y


    def soft_update(self, local_model, target_model, tau):
        """This method implements a soft update of the model parameters which connects the local and target model.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Arguments:
        =========
            local_model (PyTorch model): The weights will be copied from it.
            target_model (PyTorch model): The weights will be copied to it.
            tau (float): Sets the interpolation parameter. 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        
  

class ReplayBuffer:
    """This class defines the replay buffer of fixed size where we store the experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """We initialize a ReplayBuffer object by the constructor.

        Arguments:
        =========
            action_size (of int type): This is the dimension of each action.
            buffer_size (of int type): This is the maximum size of the buffer.
            batch_size (of int type): This gives the size of each training batch.
            seed (of int type): This sets the random seed.
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """This method adds a new experience to the memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """This method essentially takes a random sample as a batch of experiences from the memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        # The random sample is returned:
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """This returns the current size of the internal memory."""
        return len(self.memory)  
    