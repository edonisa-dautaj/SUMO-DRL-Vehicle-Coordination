import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


class DDQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DDQNNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


class DDQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, lr=1e-3,
                 epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01,
                 batch_size=64, memory_size=10000, target_update_freq=10):

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DDQNNetwork(state_size, action_size).to(self.device)
        self.target_net = DDQNNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.update_counter = 0
        self.target_update_freq = target_update_freq


    def get_action(self, state, exploit=False):
        if exploit or np.random.rand() > self.epsilon:
            state = torch.as_tensor(np.array([state]), dtype=torch.float32, device=self.device)
            with torch.no_grad():
                q_values = self.policy_net(state)
                return torch.argmax(q_values).item()
        else:
            return random.randint(0, self.action_size - 1)

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)

        #1 Select the best next action using the policy network
        next_actions = self.policy_net(next_states).argmax(1, keepdim=True)

        #2 Evaluate that action using the target network
        next_q = self.target_net(next_states).gather(1, next_actions).detach()

        target_q = rewards + (1 - dones.float()) * self.gamma * next_q

        loss = self.loss_fn(q_values, target_q)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay