import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCriticNetwork, self).__init__()

        # Shared layers used by both actor and critic
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Actor: gives action probabilities
        self.actor = nn.Sequential(
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )

        # Critic: estimates the value of the current state
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        features = self.shared_layers(x)

        action_probs = self.actor(features)
        state_value = self.critic(features)

        return action_probs, state_value


class PPOAgent:
    def __init__(
        self,
        state_size,
        action_size,
        gamma=0.99,
        lr=3e-4,
        clip_epsilon=0.2,
        ppo_epochs=4,
        value_coef=0.1,
        entropy_coef=0.02,
        debug=False
    ):
        self.state_size = state_size
        self.action_size = action_size

        # PPO parameters
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        # Debug options
        self.debug = debug
        self.debug_counter = 0
        self.debug_print_every = 100

        # Memory for one trajectory / episode
        self.memory = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = ActorCriticNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_loss_fn = nn.MSELoss()

    def state_to_tensor(self, state):
        # The environment returns a discrete state number.
        # I convert it to a one-hot vector so the neural network can process it correctly.
        state_vector = np.zeros(self.state_size, dtype=np.float32)

        # Safety check in case state is not a normal integer
        state = int(state)

        if state < 0 or state >= self.state_size:
            raise ValueError(
                f"Invalid state {state}. Expected state between 0 and {self.state_size - 1}."
            )

        state_vector[state] = 1.0

        # Convert state vector to PyTorch tensor
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)

        return state_tensor

    def get_action(self, state, exploit=False):
        # Convert state before passing it to the neural network
        state_tensor = self.state_to_tensor(state)

        # The policy outputs probabilities for each action
        # During training, the action is sampled to allow exploration
        with torch.no_grad():
            action_probs, value = self.policy(state_tensor)

        distribution = Categorical(action_probs)
        if exploit:
            # During validation, choose the action with the highest probability
            action = torch.argmax(action_probs, dim=-1)
        else:
            # During training, sample from the probability distribution
            action = distribution.sample()

        # Store the log probability of the selected action
        # PPO uses this later to compare the old policy with the updated policy
        log_prob = distribution.log_prob(action)

        if self.debug and self.debug_counter % self.debug_print_every == 0:
            probs = action_probs.detach().cpu().numpy()[0]
            print(
                f"[PPO DEBUG] state={state} | "
                f"probs={np.round(probs, 4)} | "
                f"selected_action={action.item()} | "
                f"value={value.item():.4f}"
            )

        self.debug_counter += 1

        return action.item(), log_prob.item(), value.item()

    def store(self, state, action, reward, done, log_prob, value):
        # Save one transition from the current episode
        # These values are needed later for the PPO update
        self.memory.append((state, action, reward, done, log_prob, value))

    def compute_returns_and_advantages(self):
        states, actions, rewards, dones, old_log_probs, values = zip(*self.memory)

        returns = []
        discounted_return = 0

        # Compute the discounted return for each ste
        # We go backwards because each return depends on the future rewards
        # If the episode is done, the return starts again from 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_return = 0

            discounted_return = reward + self.gamma * discounted_return
            returns.insert(0, discounted_return)

        returns = torch.FloatTensor(returns).to(self.device)
        values = torch.FloatTensor(values).to(self.device)

        # Normalize returns to make the critic update more stable
        # This helps to avoid very large or very small target values
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

        # Advantage shows if the action was better or worse than expected
        # So it compares real return with the value predicted by the critic
        # Positive advantage means the action was better than expected
        advantages = returns - values

        # Normalize advantages for stable PPO updates
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )

        return returns, advantages

    def update(self):
        if len(self.memory) == 0:
            return

        states, actions, rewards, dones, old_log_probs, values = zip(*self.memory)

        returns, advantages = self.compute_returns_and_advantages()

        states_tensor = torch.cat(
            [self.state_to_tensor(state) for state in states]
        ).to(self.device)

        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)

        last_actor_loss = 0
        last_critic_loss = 0
        last_entropy = 0
        last_total_loss = 0

        for _ in range(self.ppo_epochs):
            action_probs, state_values = self.policy(states_tensor)

            distribution = Categorical(action_probs)
            new_log_probs = distribution.log_prob(actions_tensor)
            entropy = distribution.entropy().mean()

            # PPO ratio compares the probability of the action under the new policy
            # with the probability under the old policy
            # If the ratio is close to 1, the policy didn't change too much.
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)

            # PPO clipped objective
            # It increases the probability of actions with positive advantage
            # and decreases it for actions with negative advantage.
            unclipped_objective = ratio * advantages

            # Limits how much the policy is allowed to change.
            # This is the main PPO idea because it prevents unstable big updates.
            clipped_objective = torch.clamp(
                ratio,
                1 - self.clip_epsilon,
                1 + self.clip_epsilon
            ) * advantages

            # Actor loss usess the smaller objective
            # This makes the update more conservative and stable
            actor_loss = -torch.mean(
                torch.min(unclipped_objective, clipped_objective)
            )

            state_values = state_values.squeeze(-1)

            # Critic loss checks how close the critic value prediction is to the computed return
            critic_loss = self.value_loss_fn(state_values, returns)

            # Total PPO loss
            # Total PPO loss combines three parts:
            # actor_loss updates the policy,
            # critic_loss improves the value estimation,
            # entropy encourages some exploration.
            # Entropy is subtracted because we want to keep the action probabilities
            # from becoming too deterministic too early.
            total_loss = (
                actor_loss
                + self.value_coef * critic_loss
                - self.entropy_coef * entropy
            )

            self.optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping helps avoid unstable updates
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)

            self.optimizer.step()

            last_actor_loss = actor_loss.item()
            last_critic_loss = critic_loss.item()
            last_entropy = entropy.item()
            last_total_loss = total_loss.item()

        print(
            f"[PPO UPDATE] memory_size={len(self.memory)} | "
            f"actor_loss={last_actor_loss:.4f} | "
            f"critic_loss={last_critic_loss:.4f} | "
            f"entropy={last_entropy:.4f} | "
            f"total_loss={last_total_loss:.4f}"
        )

        # Clear the trajectory buffer after the PPO update.
        # It only stores temporary experience from the current episode.
        # The learning is not lost here, because the update already changed
        # the neural network weights
        # The next episode should collect new data from the updated policy
        self.memory = []

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self.policy.eval()