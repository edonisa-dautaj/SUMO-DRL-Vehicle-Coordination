import gymnasium as gym
import sumolib
import traci
import numpy as np
from gymnasium import spaces
from Collisions import Collisions_veh


class SumoGymEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, sumo_cmd, vehicle_id, edges_list, color, max_speed=36.0):
        super(SumoGymEnv, self).__init__()

        self.use_gui = True
        self.sumo_configuration_path = sumo_cmd[2]
        self.sumo_cmd = sumo_cmd
        self.vehicle_id = vehicle_id
        self.edges_list = edges_list
        self.color = color
        self.max_speed = max_speed
        self.action_max_per_episode = 7000
        self.action_per_episode = 0

        # Espace d'observation : Nombre d'états discrets (5 colors)
        self.observation_space = spaces.Discrete(5)

        # Espace d'action : Accélérer (0), ou Ne rien faire (1), ou Ralentir (2)
        self.action_space = spaces.Discrete(3)

    def _goal_distance(self):
        speed = traci.vehicle.getSpeed(self.vehicle_id)
        return 2 * speed

    def _goal_axis_achieved(self):
        x, y = traci.vehicle.getPosition(self.vehicle_id)

        if self.vehicle_id == 'v_red':
            achieved = y <= -35
        elif self.vehicle_id == 'v_blue':
            achieved = x >= 60
        elif self.vehicle_id == 'v_orange':
            achieved = x >= 50
        else:
            print("choisis le goal axis du nv vehicule")
            achieved = False
        return achieved

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

    def get_state(self):
        achieved = self._goal_axis_achieved()
        speed = traci.vehicle.getSpeed(self.vehicle_id)
        leader_info = traci.vehicle.getLeader(self.vehicle_id)  # its ID and distance (sans mingap)
        if leader_info:
            gap = leader_info[1]  # Distance to the leader vehicle
        else:
            gap = float('inf')  # No leader in range
        safe_distance = self._goal_distance()
        return self.discretize_state(gap, safe_distance, speed, self.vehicle_id, achieved)

    @staticmethod
    def discretize_state(gap, safe_distance, speed, vehicle_id, achieved):
        # five-states : black, rouge, orange, vert, blanc
        collisions = Collisions_veh()
        min_speed = 8.33  # 30km/h
        max_speed = 25  # 90km/h
        if (vehicle_id in collisions.detect_collisions_in_junction(0.5)) or (
                vehicle_id in collisions.detect_collisions()) or (gap <= 0.3) :  # Black collision happend
            #print(traci.vehicle.getLeader(vehicle_id))
            print("black gap", gap, speed)
            return 0
        elif achieved == True and (vehicle_id not in collisions.collided_vehicles) and (vehicle_id in traci.vehicle.getIDList()) and speed > min_speed:
            #print("white")
            return 4
        elif (gap < safe_distance - 10) or (speed > max_speed) : #or (speed <= min_speed and gap >= safe_distance + 10) :  # riskyy navigation veh too close or vehcle stops and no veh in front of it
            #print("Red", speed)
            return 1  # Red
        elif (speed < min_speed and gap <= safe_distance) or (gap > safe_distance and speed >= min_speed) : #and min_speed <= speed <= max_speed):  # safe navigation or vehicule stops when its near the leader
            #print("green", speed)
            return 3  # Vert
        else:
            #print("orange", speed)
            return 2  # Orange

    def reset(self, seed=None, options=None):
        traci.start(self.sumo_cmd)
        # Réinitialiser v=10 t=0
        self.action_per_episode = 0 # init action compteur
        traci.simulationStep()
        traci.vehicle.setSpeed(self.vehicle_id, 0)
        # Obs initiale
        state = 3 #self.get_state()
        return state, {}

    def apply_action (self, action) :
        if action == 0:  # Accélérer
            traci.vehicle.setAcceleration(self.vehicle_id, min(traci.vehicle.getAcceleration(self.vehicle_id) + 1., traci.vehicle.getAccel(self.vehicle_id)), 0.1)

        elif action == 1:
            pass

        elif action == 2:  # Ralentir
            traci.vehicle.setAcceleration(self.vehicle_id, max(traci.vehicle.getAcceleration(self.vehicle_id) - 2.,-traci.vehicle.getDecel(self.vehicle_id)), 0.1)

    def step(self, action):
        done = False
        truncated = False

        # Vérifier si l'épisode a dépassé le nb d'action max
        if self.action_per_episode >= self.action_max_per_episode :
            print(f"Épisode terminé après {self.action_max_per_episode} actions.")
            state = 0
            reward = -100
            #done=True
            truncated=True
            #traci.vehicle.remove(self.vehicle_id)  # Retirer le véhicule
            self.action_per_episode = 0
            return state, reward, done, truncated, {}  # Retourner l'état noir et done=True

        # Appliquer l'action
        else :
            self.apply_action(action)
            self.action_per_episode += 1
            traci.simulationStep()  # Avancer la simulation

        if self.vehicle_id not in traci.vehicle.getIDList():
            print(f"Le véhicule {self.vehicle_id} a quitté la simulation.")
            done = True
            state = 2
            reward = 0
            #truncated = True

        else :
            # Récupérer la nouvelle vitesse
            state = self.get_state()
            if state == 0:  # État noir - collision
                reward = -20
                done = False # déjà testé de le considerer comme état final
                #traci.vehicle.remove(self.vehicle_id)  # Retirer le véhicule
            elif state == 4:  # État blanc - objectif atteint
                reward = 50
                done = False
                #traci.vehicle.remove(self.vehicle_id)  # Retirer le véhicule
            else:  # Autres états
                if state == 1:
                    reward = -5  # red
                elif state == 2:
                    reward = -1  # orange
                else:
                    reward = 5  # green
                done = False

        return state, reward, done, truncated, {}

    def render(self):
        pass  # interface graphique directement avec SUMO-gui

    def close(self):
        traci.close()
