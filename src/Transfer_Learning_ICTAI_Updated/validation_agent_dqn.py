import numpy as np
import traci
import torch
from sumo_gym_env_dqn import SumoGymEnv
from dqn import DQNAgent
from metrics import Metrics
from Collisions import Collisions_veh


def validate_dqn_agent(agent_id, sumo_config, vehicle_id, model_path, save_metric, edges_list, color, episodes=100, sumo_gui=True):
    SUMO_BINARY = "sumo-gui" if sumo_gui else "sumo"
    sumo_cmd = [
        SUMO_BINARY,
        "-c", sumo_config,
        "--collision.action", "warn",
        "--collision.check-junctions", "true",
        "--collision.mingap-factor", "0",
        "--step-length", "0.1",
        "--delay", "00",
        "--start",
        "--quit-on-end"
    ]

    env = SumoGymEnv(sumo_cmd, vehicle_id, edges_list, color)
    state, _ = env.reset()

    # Initialisation de l'agent DQN (même architecture qu'à l'entraînement)
    #sample_state, _ = env.reset()
    agent = DQNAgent(state_size=3, action_size=3)
    agent.policy_net.load_state_dict(torch.load(model_path))
    agent.policy_net.eval()

    metrics = Metrics()

    for episode in range(episodes):
        print(f"Validation Agent {agent_id} - Épisode {episode}")

        if vehicle_id not in traci.vehicle.getIDList():
            try :
                route_id = f"route_{vehicle_id}"
                if route_id not in traci.route.getIDList():
                    traci.route.add(route_id, edges_list)  # Définir la route
                traci.vehicle.add(vehicle_id, routeID=route_id, depart="now")  # Réinsérer
                traci.simulationStep()

                traci.vehicle.setColor(vehicle_id, color)
            except traci.TraCIException:
                pass


        done = False
        total_reward = 0
        step_count = 0
        first_collision_detected = False
        first_collision_step = None
        first_collision_distance = None
        episode_speeds = []
        num_affected, avg_decel = 0, 0

        init_position = traci.vehicle.getPosition(vehicle_id)
        traci.vehicle.setSpeedMode(vehicle_id, 0)
        traci.vehicle.setSpeed(vehicle_id, 10)

        state_vec = env.get_DQN_state()

        while not done:
            if vehicle_id not in traci.vehicle.getIDList():
                break

            # Exploitation seulement (pas d'exploration aléatoire)
            action = agent.get_action(state_vec, exploit=True)

            next_state, reward, done, truncated, _ = env.step(action)

            if done == True:
                break
            current_speed = traci.vehicle.getSpeed(vehicle_id)
            episode_speeds.append(current_speed)

            collisions = Collisions_veh()
            leader_info = traci.vehicle.getLeader(vehicle_id)
            gap = leader_info[1] if leader_info else float('inf')

            if not first_collision_detected and (
                vehicle_id in collisions.detect_collisions() or
                vehicle_id in collisions.detect_collisions_in_junction(0.5) or
                gap < 0 or truncated
            ):
                first_collision_detected = True
                current_position = traci.vehicle.getPosition(vehicle_id)
                dx = current_position[0] - init_position[0]
                dy = current_position[1] - init_position[1]
                first_collision_distance = (dx**2 + dy**2)**0.5
                first_collision_step = step_count

            num_affected, avg_decel = env.get_followers_deceleration(vehicle_id)

            total_reward += reward
            step_count += 1
            state_vec = env.get_DQN_state()

            if truncated:
                traci.vehicle.remove(vehicle_id)
                break

        avg_speed = sum(episode_speeds) / len(episode_speeds) if episode_speeds else 0
        success = not first_collision_detected

        metrics.update(
            total_reward,
            step_count,
            first_collision_step,
            first_collision_distance,
            success,
            avg_speed,
            num_affected,
            avg_decel
        )

    accuracy = metrics.success_per_episode
    print(f"Accuracy: {sum(accuracy) * 100 / episodes:.2f}%")

    traci.close()
    metrics.save(agent_id, save_metric)
    metrics.plot(agent_id, episodes)
    print(f"Validation terminée pour {agent_id}.")
