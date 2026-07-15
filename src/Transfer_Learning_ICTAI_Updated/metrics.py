import matplotlib.pyplot as plt
import pandas as pd

class Metrics:
    def __init__(self):
        self.episode_rewards = []
        self.episode_steps = []

        self.time_to_collision = []
        self.distance_to_collision = []

        self.average_speed_per_episode = []
        self.success_per_episode = []

        self.affected_followers = []  # Nombre de véhicules ralentis / épisode
        self.avg_deceleration = []

    def update(self, total_reward, step_count, first_collision_step=None, first_collision_distance=None, episode_success=None, average_speed=None,
               num_affected_followers=0, avg_deceleration=0):
        """
        Met à jour toutes les métriques de l'épisode.
        """
        average_reward = total_reward / step_count if step_count > 0 else 0
        travel_time = step_count * 0.1  # Conversion en secondes

        self.episode_rewards.append(average_reward)
        self.episode_steps.append(travel_time)

        # Ajoute les métriques de sécurité routière
        if first_collision_step is not None:
            self.time_to_collision.append(first_collision_step * 0.1)
        else:
            self.time_to_collision.append(None)
        self.distance_to_collision.append(first_collision_distance)

        self.success_per_episode.append(episode_success)
        self.average_speed_per_episode.append(average_speed)

        self.affected_followers.append(num_affected_followers)
        self.avg_deceleration.append(avg_deceleration)

    def save(self, agent_id, filename="metrics_agent.csv"):
        data = {
            "Episode": list(range(1, len(self.episode_rewards) + 1)),
            "Average_Reward": self.episode_rewards,
            "Travel_Time": self.episode_steps,
            "Time_to_Collision": self.time_to_collision,
            "Distance_to_Collision": self.distance_to_collision,
            "Success": self.success_per_episode,
            "Average_Speed": self.average_speed_per_episode,
            "Affected_Followers": self.affected_followers,
            "Avg_Deceleration": self.avg_deceleration
        }

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"[Agent {agent_id}] Sauvegarde des métriques terminée : {filename}")

    def plot(self, agent_id, episodes):
        """
        Affiche les graphiques pour toutes les métriques disponibles.
        """
        plt.figure(figsize=(12, 8))

        # Récompenses
        #plt.subplot(2, 3, 1)
        plt.plot(range(episodes), self.episode_rewards)
        plt.xlabel("Episodes")
        plt.ylabel("Mean Reward")
        plt.title(f"Reward - Agent {agent_id}")
        plt.grid()
        plt.show(block=False)

        # Temps de trajet
        plt.figure(figsize=(12, 8))
        #plt.subplot(2, 3, 2)
        plt.plot(range(episodes), self.episode_steps, marker='o')
        plt.xlabel("Episodes")
        plt.ylabel("travel time (s)")
        plt.title(f"travel time - Agent {agent_id}")
        plt.grid()
        plt.show(block=False)

        # Temps avant collision
        plt.figure(figsize=(12, 8))
        #plt.subplot(2, 3, 3)
        plt.plot(range(episodes), [t if t is not None else 0 for t in self.time_to_collision], marker='o')
        plt.xlabel("Episodes")
        plt.ylabel("Time to collision (s)")
        plt.title(f"Time to collision - Agent {agent_id}")
        plt.grid()
        plt.show(block=False)

        # Distance à la collision
        plt.figure(figsize=(12, 8))
        #plt.subplot(2, 3, 4)
        plt.plot(range(episodes), [d if d is not None else 0 for d in self.distance_to_collision], marker='o')
        plt.xlabel("Episodes")
        plt.ylabel("Distance to collision (m)")
        plt.title(f"Distance collision - Agent {agent_id}")
        plt.grid()
        plt.show(block=False)

        # Affichage de la métrique de succès
        plt.figure(figsize=(12, 8))
        #plt.subplot(2, 3, 5)
        plt.plot(range(episodes), self.success_per_episode, marker='o', linestyle='-')
        plt.title(f" Episode Succes - Agent {agent_id}")
        plt.xlabel("Episodes")
        plt.ylabel("Succes (1=Oui, 0=Non)")
        plt.yticks([0, 1], ["Fail", "Succes"])
        plt.grid(True)
        plt.show(block=False)

        # Vitesse moyenne
        plt.figure(figsize=(12, 8))
        #plt.subplot(2, 3, 6)
        plt.plot(range(episodes), self.average_speed_per_episode, linestyle='-', color='orange')
        plt.xlabel("Episodes")
        plt.ylabel("Mean Speed (m/s)")
        plt.title(f"Mean Speed - Agent {agent_id}")
        plt.grid()

        plt.tight_layout()
        plt.show(block=False)

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(range(episodes), self.affected_followers, marker='o', color='red')
        plt.xlabel("Episodes")
        plt.ylabel("Slowed vehicles")
        plt.title(f"Impact on traffic  - Agent {agent_id}")
        plt.grid()

        # Nouveau subplot : Décélération moyenne
        plt.subplot(2, 2, 2)
        plt.plot(range(episodes), self.avg_deceleration, color='purple')
        plt.xlabel("Episodes")
        plt.ylabel("Mean Deceleration (m/s²)")
        plt.title(f"Induced braking - Agent {agent_id}")
        plt.grid()

        plt.tight_layout()
        plt.show()

