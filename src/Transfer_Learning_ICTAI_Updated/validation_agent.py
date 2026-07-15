import numpy as np
import traci
from Collisions import Collisions_veh
from sumo_gym_env import SumoGymEnv
from q_Learning import QLearning
from perfect_q_table import PerfectQLearning
from metrics import Metrics

def validate_agent(agent_id, sumo_config, vehicle_id, q_table_file, save_metric, edges_list, color, episodes=100, sumo_gui=True):
    SUMO_BINARY = "sumo-gui" if sumo_gui else "sumo"
    sumo_cmd = [
        SUMO_BINARY,
        "-c", sumo_config,
        "--collision.action", "warn",
        "--collision.check-junctions", "true",
        "--step-length", "0.1",
        "--delay", "00",
        "--start",
        "--quit-on-end"
    ]
    env = SumoGymEnv(sumo_cmd, vehicle_id, edges_list, color)

    # Charger la Q-table entraînée
    trained_q_table = np.load(q_table_file)
    q_learning = QLearning(env.observation_space.n, env.action_space.n, q_table=trained_q_table)
    #q_learning = PerfectQLearning(env.observation_space.n, env.action_space.n)

    metrics = Metrics()
    state, _ = env.reset()

    for episode in range(episodes):
        print(f"Validation Agent {agent_id} - Épisode {episode}")

        if vehicle_id not in traci.vehicle.getIDList():
            route_id = f"route_{vehicle_id}"
            if route_id not in traci.route.getIDList():
                traci.route.add(route_id, edges_list)  # Définir la route
            traci.vehicle.add(vehicle_id, routeID=route_id, depart="now")  # Réinsérer
            traci.simulationStep()
            traci.vehicle.setColor(vehicle_id, color)

        done = False
        total_reward = 0
        step_count = 0

        traci.vehicle.setSpeedMode(vehicle_id, 0)
        traci.vehicle.setSpeed(vehicle_id, 10)
        traci.simulationStep()

        episode_speeds = []
        first_collision_detected = False
        first_collision_step = None
        first_collision_distance = None

        init_position = traci.vehicle.getPosition(vehicle_id)

        while not done:
            if vehicle_id in traci.vehicle.getIDList():
                current_speed = traci.vehicle.getSpeed(vehicle_id)
                episode_speeds.append(current_speed)

                # Choix d'action : exploitation uniquement (pas d'exploration aléatoire)
                action = np.argmax(q_learning.q_table[state])

                num_affected, avg_decel = env.get_followers_deceleration(vehicle_id)

                next_state, reward, done, truncated, _ = env.step(action)

                if done == True :
                    break

                collisions = Collisions_veh()
                leader_info = traci.vehicle.getLeader(vehicle_id)  # its ID and distance (sans mingap)
                gap = leader_info[1] if leader_info else float('inf')  # Distance to the leader vehicle

                if not first_collision_detected and ((vehicle_id in collisions.detect_collisions_in_junction(0.5)) or (
                vehicle_id in collisions.detect_collisions()) or (gap < 0)) or truncated == True:
                    first_collision_detected = True
                    current_position = traci.vehicle.getPosition(vehicle_id)
                    dx = current_position[0] - init_position[0]
                    dy = current_position[1] - init_position[1]
                    first_collision_distance = (dx**2 + dy**2)**0.5
                    first_collision_step = step_count

                total_reward += reward
                step_count += 1
                state = next_state

                if truncated == True:
                    traci.vehicle.remove(vehicle_id)
                    break

            else:
                traci.simulationStep()

        episode_success = not first_collision_detected
        average_speed = sum(episode_speeds) / len(episode_speeds) if episode_speeds else 0

        metrics.update(
            total_reward, step_count, first_collision_step,
            first_collision_distance, episode_success, average_speed,
            num_affected, avg_decel
        )
    accuracy = metrics.success_per_episode
    print("accuracy", sum(accuracy) * 100 / episodes)
    traci.close()

    # Affichage ou sauvegarde des résultats
    metrics.save(agent_id, save_metric)
    metrics.plot(agent_id, episodes)
    print(f"Validation terminée pour {agent_id}.")


