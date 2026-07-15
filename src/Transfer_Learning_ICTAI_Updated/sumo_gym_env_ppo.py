import gymnasium as gym
import traci
import numpy as np
from gymnasium import spaces
from Collisions import Collisions_veh


class SumoGymEnvPPO(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, sumo_cmd, vehicle_id, edges_list, color, max_speed=36.0):
        super(SumoGymEnvPPO, self).__init__()

        self.use_gui = True
        self.sumo_configuration_path = sumo_cmd[2]
        self.sumo_cmd = sumo_cmd
        self.vehicle_id = vehicle_id
        self.edges_list = edges_list
        self.color = color
        self.max_speed = max_speed
        self.action_max_per_episode = 7000
        self.action_per_episode = 0

        # This is used to know if SUMO was already started.
        # First reset starts SUMO, next resets only reload the same scenario.
        self.sumo_started = False

        # Observation space: 5 discrete states
        # 0 = black, 1 = red, 2 = orange, 3 = green, 4 = white
        self.observation_space = spaces.Discrete(5)

        # Action space:
        # 0 = accelerate, 1 = do nothing, 2 = decelerate
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
            print("Choose the goal axis for the new vehicle.")
            achieved = False

        return achieved

    def get_followers_deceleration(self, vehicle_id):
        """
        Returns the number of followers that are braking and their average deceleration.
        """
        followers = []
        current_leader = vehicle_id
        max_depth = 3

        for _ in range(max_depth):
            follower_info = traci.vehicle.getFollower(current_leader)

            if follower_info and follower_info[0]:
                follower_id, dist = follower_info
                followers.append(follower_id)
                current_leader = follower_id
            else:
                break

        decelerations = []

        for follower_id in followers:
            accel = traci.vehicle.getAcceleration(follower_id)

            if accel < -0.5:
                decelerations.append(abs(accel))

        return len(decelerations), np.mean(decelerations) if decelerations else 0

    def get_state(self):
        achieved = self._goal_axis_achieved()
        speed = traci.vehicle.getSpeed(self.vehicle_id)

        leader_info = traci.vehicle.getLeader(self.vehicle_id)

        if leader_info:
            gap = leader_info[1]
        else:
            gap = float('inf')

        # Extra check: distance to the closest other vehicle.
        # This helps the agent detect vehicles coming from another direction in the junction,
        # not only the leader in the same lane.
        min_other_distance = float('inf')

        if self.vehicle_id in traci.vehicle.getIDList():
            x1, y1 = traci.vehicle.getPosition(self.vehicle_id)

            for other_id in traci.vehicle.getIDList():
                if other_id != self.vehicle_id:
                    x2, y2 = traci.vehicle.getPosition(other_id)
                    distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                    if distance < min_other_distance:
                        min_other_distance = distance

        safe_distance = self._goal_distance()

        return self.discretize_state(
            gap,
            safe_distance,
            speed,
            self.vehicle_id,
            achieved,
            min_other_distance
        )

    @staticmethod
    def discretize_state(gap, safe_distance, speed, vehicle_id, achieved, min_other_distance):
        """
        Converts traffic situation into one of 5 discrete states:
        0 = black  -> collision / dangerous
        1 = red    -> risky
        2 = orange -> warning
        3 = green  -> safe
        4 = white  -> goal achieved
        """
        collisions = Collisions_veh()

        min_speed = 8.33  # 30 km/h
        max_speed = 25    # 90 km/h

        if (
            vehicle_id in collisions.detect_collisions_in_junction(0.5)
            or vehicle_id in collisions.detect_collisions()
            or gap <= 0.3
            or min_other_distance <= 5.0
        ):
            return 0

        elif (
            achieved is True
            and vehicle_id not in collisions.collided_vehicles
            and vehicle_id in traci.vehicle.getIDList()
            and speed > min_speed
        ):
            return 4

        elif gap < safe_distance - 10 or speed > max_speed:
            return 1

        elif (
            speed < min_speed and gap <= safe_distance
        ) or (
            gap > safe_distance and speed >= min_speed
        ):
            return 3

        else:
            return 2

    def reset(self, seed=None, options=None):
        """
        Starts a new SUMO episode.

        PPO version:
        - starts SUMO only the first time
        - for the next episodes, reloads the same scenario using traci.load()
        - avoids closing/opening the SUMO window after every episode
        - gives the vehicle an initial speed so it does not stay frozen
        """
        super().reset(seed=seed)

        # First episode: start SUMO normally.
        if not self.sumo_started:
            traci.start(self.sumo_cmd)
            self.sumo_started = True

            # Advance one simulation step so vehicles are loaded
            traci.simulationStep()

        # Reset action counter
        self.action_per_episode = 0

        # If the controlled vehicle already left the simulation,
        if self.vehicle_id not in traci.vehicle.getIDList():
            try:
                route_id = f"route_{self.vehicle_id}"

                # Add the route only if it does not already exist
                if route_id not in traci.route.getIDList():
                    traci.route.add(route_id, self.edges_list)

                # Reinsert the controlled vehicle into the running simulation
                traci.vehicle.add(self.vehicle_id, routeID=route_id, depart="now")
                traci.simulationStep()

            except traci.TraCIException:
                pass

        # Give initial color and speed to the controlled vehicle.
        # This keeps the vehicle visible and avoids it staying frozen at the start.
        if self.vehicle_id in traci.vehicle.getIDList():
            traci.vehicle.setColor(self.vehicle_id, self.color)
            traci.vehicle.setSpeedMode(self.vehicle_id, 0)
            traci.vehicle.setSpeed(self.vehicle_id, 0)

        state = 3

        return state, {}


    def apply_action(self, action):
        """
        Applies the action selected by the PPO agent.
        """

        # Safety check before applying any action to prevent errors
        # if the vehicle has already left the simulation
        if self.vehicle_id not in traci.vehicle.getIDList():
            return

        if action == 0:  # Accelerate
            current_acc = traci.vehicle.getAcceleration(self.vehicle_id)
            max_acc = traci.vehicle.getAccel(self.vehicle_id)

            traci.vehicle.setAcceleration(
                self.vehicle_id,
                min(current_acc + 1.0, max_acc),
                0.1
            )

        elif action == 1:  # Do nothing
            pass

        elif action == 2:  # Decelerate
            current_acc = traci.vehicle.getAcceleration(self.vehicle_id)
            max_decel = traci.vehicle.getDecel(self.vehicle_id)

            traci.vehicle.setAcceleration(
                self.vehicle_id,
                max(current_acc - 2.0, -max_decel),
                0.1
            )

    def step(self, action):
        done = False
        truncated = False

        # Stop the episode if it exceeds the maximum number of actions
        if self.action_per_episode >= self.action_max_per_episode:
            print(f"Episode finished after {self.action_max_per_episode} actions.")

            state = 0
            reward = -100
            truncated = True
            self.action_per_episode = 0

            return state, reward, done, truncated, {}

        # Apply action and advance simulation
        self.apply_action(action)
        self.action_per_episode += 1
        traci.simulationStep()

        # If the vehicle left the simulation, consider it as successful completion
        if self.vehicle_id not in traci.vehicle.getIDList():
            print(f"Vehicle {self.vehicle_id} left the simulation.")

            # if the vehicle finishes its route and leaves the simulation without collision
            # this should be considered a successful episode.
            done = True
            state = 4
            reward = 50

        else:
            state = self.get_state()

            if state == 0:
                # Black state: collision or very dangerous situation
                reward = -20
                done = False

            elif state == 4:
                # White state: goal achieved
                reward = 50
                done = False

            else:
                if state == 1:
                    # Red state: risky
                    reward = -5

                elif state == 2:
                    # Orange state: warning
                    reward = -1

                else:
                    # Green state: safe
                    reward = 5

                done = False

        return state, reward, done, truncated, {}

    def render(self):
        pass

    def close(self):
        """
        Safely closes the TraCI connection.
        """
        try:
            if traci.isLoaded():
                traci.close(False)
        except Exception:
            pass

        self.sumo_started = False