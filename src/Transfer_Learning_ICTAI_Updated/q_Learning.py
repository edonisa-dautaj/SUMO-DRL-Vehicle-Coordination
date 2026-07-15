import numpy as np
import random
import traci
from sumo_gym_env import SumoGymEnv


class QLearning:
    def __init__(self, state_size=5, action_size=3, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.99,
                 epsilon_min=0, q_table = None ):
        self.alpha = alpha  # Taux d'apprentissage
        self.gamma = gamma  # Facteur de discount
        self.epsilon = epsilon  # Exploration initiale
        self.epsilon_decay = epsilon_decay  # Réduction de l'exploration
        self.epsilon_min = epsilon_min  # Exploration minimale
        if q_table is None:
            self.q_table = np.zeros((state_size, action_size))
        else:
            self.q_table = q_table

    def update(self, state, action, reward, next_state):
        """Met à jour la Q-table """

        self.q_table[state, action] = (1 - self.alpha) * self.q_table[state, action] + \
                                      self.alpha * (reward + self.gamma * np.max(self.q_table[next_state, :]))

    def get_action(self, state, action_space):
        """Retourne l'action à prendre selon la politique epsilon-greedy."""
        if random.uniform(0, 1) < self.epsilon:
            #print("random")
            return action_space.sample()  # Exploration
        #print("Exploitation")
        return np.argmax(self.q_table[state, :])  # Exploitation

    def decay_epsilon(self, episode, total_episodes):
        if episode < 10:
            self.epsilon = 1.0
        elif episode >= total_episodes - 10:
            self.epsilon = self.epsilon_min
        else:
            decay_range = total_episodes - 20  # Nombre d’épisodes sur lesquels faire décroître epsilon
            progress = (episode - 10) / decay_range  # Progression linéaire entre 0 et 1
            self.epsilon = 1.0 - (1.0 - self.epsilon_min) * progress
