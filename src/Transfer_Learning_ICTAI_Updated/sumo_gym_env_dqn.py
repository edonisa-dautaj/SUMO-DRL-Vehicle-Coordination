from gymnasium import spaces
import numpy as np
import traci
from Collisions import Collisions_veh
import gymnasium as gym


class SumoGymEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, sumo_cmd, vehicle_id, edges_list, color, max_speed=36.0):
        super(SumoGymEnv, self).__init__()

        self.use_gui = True
        self.sumo_cmd = sumo_cmd
        self.vehicle_id = vehicle_id
        self.edges_list = edges_list
        self.color = color
        self.max_speed = max_speed
        self.action_max_per_episode = 7000
        self.action_per_episode = 0

        # Espace d'état continu : [speed, gap, waiting time]
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)

        # Actions : Accélérer, rien faire, ralentir
        self.action_space = spaces.Discrete(3)

    def get_DQN_state(self):
        speed = traci.vehicle.getSpeed(self.vehicle_id)
        leader = traci.vehicle.getLeader(self.vehicle_id)
        gap = leader[1] if leader else 100  # pas de leader proche
        waiting_time = traci.vehicle.getAccumulatedWaitingTime(self.vehicle_id)

        state = np.array([
            speed / 30.0,               # Normalisation vitesse (max = 30 m/s)
            min(gap, 50) / 50.0,        # Normalisation gap (coupé à 50 m)
            waiting_time / 100.0        # Normalisation attente
        ], dtype=np.float32)

        return state

    def get_followers_deceleration(self, vehicle_id):
        """Retourne la décélération moyenne des véhicules derrière."""
        followers = []
        current_leader = vehicle_id
        max_depth = 3  # Nombre de véhicules à analyser

        for _ in range(max_depth):
            follower_info = traci.vehicle.getFollower(current_leader)
            if follower_info and follower_info[0]:  # Si un follower existe
                follower_id, dist = follower_info
                followers.append(follower_id)
                current_leader = follower_id
            else:
                break

        decelerations = []
        for follower_id in followers:
            accel = traci.vehicle.getAcceleration(follower_id)
            if accel < -0.5:  # Seuil de freinage
                decelerations.append(abs(accel))
        return len(decelerations), np.mean(decelerations) if decelerations else 0

    def _goal_axis_achieved(self):
        x, y = traci.vehicle.getPosition(self.vehicle_id)

        if self.vehicle_id == 'v_red':
            return y <= -35
        elif self.vehicle_id == 'v_blue':
            return x >= 60
        elif self.vehicle_id == 'v_orange':
            return x >= 50
        else:
            return False

    def compute_reward(self, speed, gap, waiting_time, achieved, has_collision):
        reward = 0
        min_speed = 8.33
        max_speed = 25.0
        safe_distance = 2 * speed

        #  Cas collision = fort malus
        if has_collision or gap <= 0.3:
            return -20

        #  Cas objectif atteint = gros bonus
        if achieved and speed > min_speed:
            return 50

        # Risqué : véhicule trop proche ou trop rapide
        if (gap < safe_distance - 10) or (speed > max_speed):
            reward -= 5  # Rouge

        # Safe : bonne distance ou bon arrêt
        elif (speed < min_speed and gap <= safe_distance) or (gap > safe_distance and speed >= min_speed):
            reward += 5  # Vert

        # Cas normal
        else:
            reward -= 1  # Orange

        # ⏱ Bonus si le véhicule attend peu
        reward -= 0.01 * waiting_time

        return reward

    def reset(self, seed=None, options=None):
        traci.start(self.sumo_cmd)
        self.action_per_episode = 0
        traci.simulationStep()
        traci.vehicle.setSpeed(self.vehicle_id, 0)
        return self.get_DQN_state(), {}

    def apply_action(self, action):
        if action == 0:  # Accélérer
            traci.vehicle.setAcceleration(self.vehicle_id, min(traci.vehicle.getAcceleration(self.vehicle_id) + 1.0, traci.vehicle.getAccel(self.vehicle_id)), 0.1)
        elif action == 2:  # Ralentir
            traci.vehicle.setAcceleration(self.vehicle_id, max(traci.vehicle.getAcceleration(self.vehicle_id) - 2.0, -traci.vehicle.getDecel(self.vehicle_id)), 0.1)
        else:
            pass

    def step(self, action):
        done = False
        truncated = False

        if self.action_per_episode >= self.action_max_per_episode:
            truncated = True
            self.action_per_episode = 0
            return self.get_DQN_state(), -100, done, truncated, {}

        self.apply_action(action)
        self.action_per_episode += 1
        traci.simulationStep()

        if self.vehicle_id not in traci.vehicle.getIDList():
            return np.zeros(3, dtype=np.float32), 0, True, truncated, {}

        # Récupération de l'état
        state = self.get_DQN_state()

        # Reward logique similaire à l'ancien système
        speed = traci.vehicle.getSpeed(self.vehicle_id)
        leader = traci.vehicle.getLeader(self.vehicle_id)
        gap = leader[1] if leader else 100
        waiting_time = traci.vehicle.getAccumulatedWaitingTime(self.vehicle_id)
        achieved = self._goal_axis_achieved()
        has_collision = (self.vehicle_id in Collisions_veh().detect_collisions()) or gap < 0.3 or (self.vehicle_id in Collisions_veh().detect_collisions_in_junction(0.5))

        reward = self.compute_reward(speed, gap, waiting_time, achieved, has_collision)

        return state, reward, done, truncated, {}

    def render(self):
        pass

    def close(self):
        traci.close()
