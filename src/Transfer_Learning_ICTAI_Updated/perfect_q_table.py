import numpy as np
import random
import traci
from sumo_gym_env import SumoGymEnv


class PerfectQLearning:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.9, epsilon=0, epsilon_decay=0.95,
                 epsilon_min=0.01):
        self.alpha = alpha  # Taux d'apprentissage
        self.gamma = gamma  # Facteur de discount
        self.epsilon = epsilon  # Exploration initiale
        self.epsilon_decay = epsilon_decay  # Réduction de l'exploration
        self.epsilon_min = epsilon_min  # Exploration minimale
        self.q_table = np.array([[0, 0, 10.],
       [0., 1., 20.],
       [70, 50., 60.],
       [90., 80., 70.],
       [100., 90., 80.]])

    def update(self, state, action, reward, next_state):
        """Met à jour la Q-table selon l'équation de Bellman."""
        """
        self.q_table[state, action] = (1 - self.alpha) * self.q_table[state, action] + \
                                      self.alpha * (reward + self.gamma * np.max(self.q_table[next_state, :]))
        """
        pass

    def get_action(self, state, action_space):
        """Retourne l'action à prendre selon la politique epsilon-greedy."""
        if random.uniform(0, 1) < self.epsilon:
            return action_space.sample()  # Exploration
        return np.argmax(self.q_table[state, :])  # Exploitation

    def decay_epsilon(self):
        """Réduit epsilon progressivement."""
        #self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.epsilon = 0
        #print(self.epsilon)
