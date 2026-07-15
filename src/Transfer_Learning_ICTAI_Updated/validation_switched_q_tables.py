import numpy as np
import traci
from q_Learning import QLearning
from sumo_gym_env import SumoGymEnv

def simulation_switched_q_tables(sumo_config, vehicle_1, vehicle_2, q_table_1_file, q_table_2_file, steps=500, sumo_gui=True):
    SUMO_BINARY = "sumo-gui" if sumo_gui else "sumo"
    sumo_cmd = [SUMO_BINARY, "-c", sumo_config, "--step-length", "0.001", "--start", "--quit-on-end"]

    # Environnement partagé
    env = SumoGymEnv(sumo_cmd, vehicle_id=vehicle_1, edges_list=('-E0', 'E1'), color=(0, 0, 255, 255))

    # Charger les Q-tables
    q_table_1 = np.load(q_table_1_file)
    q_table_2 = np.load(q_table_2_file)

    # Créer les agents avec leurs Q-tables respectives
    q_learning_1 = QLearning(env.observation_space.n, env.action_space.n, q_table=q_table_2)  # véhicule_1 utilise q_table_2
    q_learning_2 = QLearning(env.observation_space.n, env.action_space.n, q_table=q_table_1)  # véhicule_2 utilise q_table_1

    # Démarrer l’environnement
    state_1, _ = env.reset()

    # Ajouter manuellement le second véhicule
    if vehicle_2 not in traci.vehicle.getIDList():
        traci.route.add("route2", ('-E0', 'E1'))
        traci.vehicle.add(vehicle_2, routeID="route2", typeID="car", depart="now")
        traci.vehicle.setColor(vehicle_2, (255, 0, 0, 255))
        traci.vehicle.setSpeedMode(vehicle_2, 0)
        traci.vehicle.setSpeed(vehicle_2, 10)

    traci.vehicle.setSpeedMode(vehicle_1, 0)
    traci.vehicle.setSpeed(vehicle_1, 10)

    print("🔄 Q-tables échangées ! Simulation en cours...")

    for step in range(steps):
        print(f"🔹 Étape {step}/{steps}")

        # VEHICULE 1
        if vehicle_1 in traci.vehicle.getIDList():
            action_1 = q_learning_1.get_action(state_1, env.action_space)
            speed_1 = traci.vehicle.getSpeed(vehicle_1)
            if action_1 == 0:
                traci.vehicle.setSpeed(vehicle_1, max(speed_1 - 1., 0.))
            elif action_1 == 1:
                traci.vehicle.setSpeed(vehicle_1, speed_1)
            elif action_1 == 2:
                traci.vehicle.setSpeed(vehicle_1, min(speed_1 + 1., env.max_speed))
            # tu peux mettre ici un vrai env.step si tu veux les rewards/env dynamics

        # VEHICULE 2
        if vehicle_2 in traci.vehicle.getIDList():
            action_2 = q_learning_2.get_action(state_1, env.action_space)  # même état pour simplifier
            speed_2 = traci.vehicle.getSpeed(vehicle_2)
            if action_2 == 0:
                traci.vehicle.setSpeed(vehicle_2, max(speed_2 - 1., 0.))
            elif action_2 == 1:
                traci.vehicle.setSpeed(vehicle_2, speed_2)
            elif action_2 == 2:
                traci.vehicle.setSpeed(vehicle_2, min(speed_2 + 1., env.max_speed))

        traci.simulationStep()

    traci.close()
    print("✅ Simulation terminée des switched Q-tables !")
