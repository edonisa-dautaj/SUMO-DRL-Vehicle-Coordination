import numpy as np
import traci

from Collisions import Collisions_veh
from perfect_q_table import PerfectQLearning
from q_Learning import QLearning
from sumo_gym_env import SumoGymEnv
from metrics import Metrics


def train_with_transfer(agent_id, sumo_config, vehicle_id, own_q_file, other_q_file, save_file, save_metric, edges_list, color, episodes=100,
                        sumo_gui=True):
    SUMO_BINARY = "sumo-gui" if sumo_gui else "sumo"
    sumo_cmd = [
        SUMO_BINARY,
        "-c", sumo_config,
        "--collision.action", "warn",
        "--collision.check-junctions", "true",
        "--step-length", "0.1",
        "--delay", "0",
        "--start",
        "--quit-on-end"
    ]

    env = SumoGymEnv(sumo_cmd, vehicle_id, edges_list, color)
    total_actions_done = 0  # nombre total d'actions faites par l'agent

    # Charger Q-table de v_1 (à finetune)
    own_q_table = np.load(own_q_file)
    own_agent = QLearning(env.observation_space.n, env.action_space.n, q_table=own_q_table)
    #own_agent = PerfectQLearning(env.observation_space.n, env.action_space.n)

    # Charger Q-table de orange maitre (guide)
    other_q_table = np.load(other_q_file)

    metrics = Metrics()
    state, _ = env.reset()

    for episode in range(episodes):

        print(f"[Transfert]", f"Agent {agent_id}", f"Épisode {episode}")

        if vehicle_id not in traci.vehicle.getIDList():
            try:
                route_id = f"route_{vehicle_id}"
                if route_id not in traci.route.getIDList():
                    traci.route.add(route_id, edges_list)  # Définir la route
                traci.vehicle.add(vehicle_id, routeID=route_id, depart="now")  # Réinsérer
                traci.simulationStep()

                traci.vehicle.setColor(vehicle_id, color)
            except traci.TraCIException:
                pass

        traci.vehicle.setSpeedMode(vehicle_id, 0)
        done = False

        first_collision_detected = False
        first_collision_step = None
        first_collision_distance = None

        total_reward = 0
        step_count = 0
        init_position = traci.vehicle.getPosition(vehicle_id)
        episode_speeds = []

        # Calcul du facteur de confiance dégressif (au début on fait + confiance à v1)
        #trust_weight = max(0.1, 1.0 - (episode / episodes))  # diminue de 1.0 → 0.1
        trust_weight = own_agent.epsilon

        while not done:
            if vehicle_id in traci.vehicle.getIDList():
                current_speed = traci.vehicle.getSpeed(vehicle_id)
                episode_speeds.append(current_speed)

                num_affected, avg_decel = env.get_followers_deceleration(vehicle_id)

                action = np.argmax(own_q_table[state])
                #action = np.argmax(other_q_table[state]) # Ancien (pas d'exploration)
                #action = own_agent.get_action(state, env.action_space) # (epsilon-greedy)
                next_state, reward, done, truncated, _ = env.step(action)
                total_actions_done += 1

                if done == True :
                    break

                collisions = Collisions_veh()
                leader_info = traci.vehicle.getLeader(vehicle_id)  # its ID and distance (sans mingap)
                gap = leader_info[1] if leader_info else float('inf')  # Distance to the leader vehicle

                if not first_collision_detected and ((vehicle_id in collisions.detect_collisions_in_junction(0.5)) or (
                vehicle_id in collisions.detect_collisions()) or (gap < 0)) : #next_state == 0:
                    first_collision_detected = True
                    current_position = traci.vehicle.getPosition(vehicle_id)
                    dx = current_position[0] - init_position[0]
                    dy = current_position[1] - init_position[1]
                    total_distance_travelled = (dx ** 2 + dy ** 2) ** 0.5

                    first_collision_step = step_count
                    first_collision_distance = total_distance_travelled

                total_reward += reward
                step_count += 1

                # Mise à jour personnalisée avec pondération
                old_value = own_agent.q_table[state, action]
                #learned_value = reward + own_agent.gamma * np.max(own_agent.q_table[next_state])

                # Pondérer avec la Q-table guide
                guided_value = other_q_table[state, action]
                """
                new_value = (1 - own_agent.alpha) * old_value + own_agent.alpha * (
                        trust_weight * guided_value + (1 - trust_weight) * learned_value
                )
                """
                new_value = trust_weight * guided_value + (1-trust_weight) * old_value
                print("trust_weight", trust_weight)
                own_agent.q_table[state, action] = new_value

                state = next_state

            else:

                traci.simulationStep()

        episode_success = not first_collision_detected
        average_speed = sum(episode_speeds) / len(episode_speeds) if episode_speeds else 0
        metrics.update(total_reward, step_count, first_collision_step, first_collision_distance, episode_success, average_speed, num_affected, avg_decel)
        own_agent.decay_epsilon(episode, episodes)
    print(own_agent.q_table)
    traci.close()

    print(f"Nombre total d'actions effectuées        : {total_actions_done}")

    # Sauvegarde de la nouvelle Q-table raffinée
    np.save(save_file, own_agent.q_table)
    print("[Transfert] Q-table affinée sauvegardée ")
    metrics.save(agent_id, save_metric)
    metrics.plot(agent_id, episodes)
