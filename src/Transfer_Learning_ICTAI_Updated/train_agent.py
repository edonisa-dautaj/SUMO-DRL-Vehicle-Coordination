
import numpy as np
import traci
from Collisions import Collisions_veh
from perfect_q_table import PerfectQLearning
from q_Learning import QLearning
from sumo_gym_env import SumoGymEnv
from metrics import Metrics


def train_agent(agent_id, sumo_config, vehicle_id, save_file, save_metric, edges_list, color, episodes=100, sumo_gui=False):
    """
    Fonction pour entraîner un agent avec des paramètres personnalisés.

    :param agent_id: Identifiant de l'agent
    :param sumo_config: Fichier de configuration SUMO
    :param vehicle_id: ID du véhicule contrôlé
    :param save_file: Nom du fichier pour sauvegarder la Q-table
    :param episodes: Nombre d'épisodes d'entraînement
    :param sumo_gui: Affichage SUMO GUI (True) ou en mode headless (False)
    """

    # Définir la commande SUMO
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

    #initial values
    step_length = 0.1  # même valeur que dans sumo_cmd
    total_actions_done = 0  # nombre total d'actions faites par l'agent
    env = SumoGymEnv(sumo_cmd, vehicle_id, edges_list, color)

    #q_table_red_trained = np.load("./models/ICTAI/q_table_agent_red_ICTAI.npy")
    q_table_orange_trained = np.load("./models/ICTAI/q_table_agent_orange_ICTAI_2.npy")
    q_learning = QLearning(env.observation_space.n, env.action_space.n)
    #q_learning = PerfectQLearning(env.observation_space.n, env.action_space.n)

    metrics = Metrics()
    state, _ = env.reset()

    init_position = traci.vehicle.getPosition(vehicle_id)

    for episode in range(episodes):
        print(f"Agent {agent_id} - Épisode {episode} ")

        traci.simulationStep()

        done = False
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

        first_collision_detected = False
        first_collision_step = None
        first_collision_distance = None

        total_reward = 0
        step_count = 0

        traci.vehicle.setSpeedMode(vehicle_id, 0)
        traci.vehicle.setSpeed(vehicle_id, 10)
        traci.simulationStep()

        episode_speeds = []

        while not done:
            if vehicle_id in traci.vehicle.getIDList():
                current_speed = traci.vehicle.getSpeed(vehicle_id)
                episode_speeds.append(current_speed)

                num_affected, avg_decel = env.get_followers_deceleration(vehicle_id)
                # print(num_affected, avg_decel)

                action = q_learning.get_action(state, env.action_space)
                next_state, reward, done, truncated, _ = env.step(action)
                total_actions_done += 1

                if done == True :
                    break

                collisions = Collisions_veh()
                leader_info = traci.vehicle.getLeader(vehicle_id)  # its ID and distance (sans mingap)
                gap = leader_info[1] if leader_info else float('inf')  # Distance to the leader vehicle

                if not first_collision_detected and ((vehicle_id in collisions.detect_collisions_in_junction(0.5)) or (
                vehicle_id in collisions.detect_collisions()) or (gap < 0)) or truncated == True : #next_state == 0:
                    first_collision_detected = True
                    current_position = traci.vehicle.getPosition(vehicle_id)
                    dx = current_position[0] - init_position[0]
                    dy = current_position[1] - init_position[1]
                    total_distance_travelled = (dx ** 2 + dy ** 2) ** 0.5

                    first_collision_step = step_count
                    first_collision_distance = total_distance_travelled

                total_reward += reward
                step_count += 1

                q_learning.update(state, action, reward, next_state)
                state = next_state

                if truncated == True:
                    traci.vehicle.remove(vehicle_id)
                    break

            else:
                traci.simulationStep()
                step_count += 1

        episode_success = not first_collision_detected
        average_speed = sum(episode_speeds) / len(episode_speeds) if episode_speeds else 0

        metrics.update(
            total_reward, step_count, first_collision_step,
            first_collision_distance, episode_success, average_speed,
           num_affected, avg_decel
        )

        q_learning.decay_epsilon(episode, total_episodes=episodes)

        if episode % 10 == 0:
            print(f"Agent {agent_id} - Épisode {episode}, Exploration {q_learning.epsilon:.2f}")

    traci.close()

    #action_time = total_actions_done * step_length

    print(f"Nombre total d'actions effectuées        : {total_actions_done}")
    #print(f"Temps total simulé selon actions         : {action_time:.2f} s")


    # Sauvegarde de la Q-table
    print(q_learning.q_table)
    np.save(save_file, q_learning.q_table)
    print(f"Agent {agent_id} - Entraînement terminé ! Q-table sauvegardée sous {save_file}")

    accuracy = metrics.success_per_episode
    print("Accuracy", sum(accuracy) * 100 / episodes)

    # Plots des résultats
    metrics.save(agent_id, save_metric)
    metrics.plot(agent_id, episodes)
