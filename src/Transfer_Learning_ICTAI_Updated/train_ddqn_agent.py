from ddqn import DDQNAgent
from sumo_gym_env_dqn import SumoGymEnv
import numpy as np
import traci
from metrics import Metrics
import torch
from Collisions import Collisions_veh  # Added for collision detection



def train_ddqn(agent_id, sumo_config, vehicle_id, save_model, save_metric, edges_list, color, episodes=1000, sumo_gui=False):
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
    env.reset()
    agent = DDQNAgent(state_size=3, action_size=3)
    metrics = Metrics()

    init_position = traci.vehicle.getPosition(vehicle_id)

    for episode in range(episodes):
        print(f"Agent {agent_id} - Épisode {episode}")
        traci.simulationStep()

        # Reinsert the vehicle if it no longer exists
        if vehicle_id not in traci.vehicle.getIDList():
            try:
                route_id = f"route_{vehicle_id}"
                if route_id not in traci.route.getIDList():
                    traci.route.add(route_id, edges_list)
                traci.vehicle.add(vehicle_id, routeID=route_id, depart="now")
                traci.simulationStep()
                traci.vehicle.setColor(vehicle_id, color)
            except traci.TraCIException:
                pass

        traci.vehicle.setSpeedMode(vehicle_id, 0)

        state_vec = env.get_DQN_state()
        done = False
        truncated = False
        total_reward = 0
        step_count = 0
        episode_speeds = []

        # Specific metrics
        first_collision_detected = False
        first_collision_step = None
        first_collision_distance = None
        num_affected = None
        avg_decel = None
        collisions = Collisions_veh()

        while not done and not truncated:
            if vehicle_id not in traci.vehicle.getIDList():
                break

            action = agent.get_action(state_vec)
            next_state, reward, done, truncated, _ = env.step(action)

            if done:
                break

            current_speed = traci.vehicle.getSpeed(vehicle_id)
            episode_speeds.append(current_speed)

            # Collision or minimum distance exceeded
            leader_info = traci.vehicle.getLeader(vehicle_id)
            gap = leader_info[1] if leader_info else float('inf')

            if not first_collision_detected and (
                vehicle_id in collisions.detect_collisions()
                or vehicle_id in collisions.detect_collisions_in_junction(0.5)
                or gap < 0.3
                or truncated
            ):
                first_collision_detected = True
                current_position = traci.vehicle.getPosition(vehicle_id)
                dx = current_position[0] - init_position[0]
                dy = current_position[1] - init_position[1]
                first_collision_distance = (dx**2 + dy**2)**0.5
                first_collision_step = step_count

            # Followers' deceleration
            num_affected, avg_decel = env.get_followers_deceleration(vehicle_id)

            next_state_vec = env.get_DQN_state()
            agent.store(state_vec, action, reward, next_state_vec, done or truncated)
            agent.train_step()

            state_vec = next_state_vec
            total_reward += reward
            step_count += 1

        agent.decay_epsilon()

        success = not first_collision_detected
        avg_speed = np.mean(episode_speeds) if episode_speeds else 0

        # Sauvegarde des métriques étendues
        metrics.update(
            total_reward, step_count, first_collision_step,
            first_collision_distance, success, avg_speed,
            num_affected, avg_decel
        )

        print(f"Épisode {episode} - Total reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    traci.close()

    torch.save(agent.policy_net.state_dict(), save_model)
    print(f"DDQN model saved: {save_model}")

    metrics.save(agent_id, save_metric)
    metrics.plot(agent_id, episodes)