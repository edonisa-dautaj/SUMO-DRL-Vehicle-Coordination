from ppo import PPOAgent
from sumo_gym_env_ppo import SumoGymEnvPPO
import numpy as np
import traci
from metrics import Metrics
from Collisions import Collisions_veh


def validate_ppo_agent(agent_id, sumo_config, vehicle_id, model_path,
                       save_metric, edges_list, color, episodes=100, sumo_gui=True):

    # SUMO configuration
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

    # Create SUMO environment for PPO validation
    env = SumoGymEnvPPO(sumo_cmd, vehicle_id, edges_list, color)

    # Initialize PPO agent with the same state and action sizes used during training
    agent = PPOAgent(
        state_size=env.observation_space.n,
        action_size=env.action_space.n,
        debug=False
    )

    # Load the trained PPO model.
    # In validation we do not train again, we only test the learned policy.
    agent.load(model_path)

    metrics = Metrics()

    # Validation loop
    for episode in range(episodes):
        print(f"Validation Agent {agent_id} - Épisode {episode}")

        # Reset environment at the beginning of each validation episode.
        # SUMO is not closed here anymore.
        # The environment reset should reload the scenario using traci.load(),
        # so the simulation starts fresh without reopening the SUMO window every time.
        state, _ = env.reset()

        # Skip episode if the controlled vehicle is not loaded
        if vehicle_id not in traci.vehicle.getIDList():
            continue

        init_position = traci.vehicle.getPosition(vehicle_id)

        # For the red intersection scenario, the vehicle needs a short initial push
        # to avoid staying stopped at the beginning of validation. After one step,
        # the fixed speed is released so PPO actions can control the vehicle.
        # For the orange merge scenario, this release caused the vehicle to move
        # with a very high default speed, so the environment reset setup is kept.
        if vehicle_id == "v_red":
            traci.vehicle.setSpeedMode(vehicle_id, 0)
            traci.vehicle.setSpeed(vehicle_id, 8)
            traci.simulationStep()
            traci.vehicle.setSpeed(vehicle_id, -1)

        done = False
        terminated = False
        truncated = False

        total_reward = 0
        step_count = 0
        episode_speeds = []

        # Variables used for collision and traffic metrics
        first_collision_detected = False
        first_collision_step = None
        first_collision_distance = None
        num_affected = None
        avg_decel = None
        collisions = Collisions_veh()

        # Run one validation episode
        while not done:
            # Stop episode if the controlled vehicle leaves the simulation
            if vehicle_id not in traci.vehicle.getIDList():
                break

            # During validation, exploit=True means that the agent always chooses
            # the action with the highest probability.
            # This removes randomness, so we can test the learned policy more clearly.

            action, _, _ = agent.get_action(state, exploit=False)

            # print("step:", step_count, "action:", action, "speed:", traci.vehicle.getSpeed(vehicle_id))

            # Apply action in the SUMO environment.
            # The environment returns the next state, reward, and information
            # about whether the episode finished normally or was stopped.
            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            # Detect if SUMO teleported the vehicle because it was stuck too long.
            # I count this as a failure, because the agent did not complete the episode normally.
            teleported = (
                vehicle_id in traci.simulation.getStartingTeleportIDList()
                or vehicle_id in traci.simulation.getEndingTeleportIDList()
            )

            if teleported:
                first_collision_detected = True
                first_collision_step = step_count
                first_collision_distance = None
                done = True

            vehicle_exists = vehicle_id in traci.vehicle.getIDList()

            if vehicle_exists:
                current_speed = traci.vehicle.getSpeed(vehicle_id)
                episode_speeds.append(current_speed)

                '''
                # If the vehicle is almost not moving for a long time, count it as failure
                if step_count > 500 and np.mean(episode_speeds[-100:]) < 0.1:
                    first_collision_detected = True
                    first_collision_step = step_count
                    first_collision_distance = None
                    done = True
                '''

                # Check distance to leader vehicle
                leader_info = traci.vehicle.getLeader(vehicle_id)
                gap = leader_info[1] if leader_info else float("inf")

                # Check if a collision or dangerous situation happened.
                # I only save the first collision, because the metrics need the first step
                # and distance where the failure happened.
                if not first_collision_detected and (
                    vehicle_id in collisions.detect_collisions()
                    or vehicle_id in collisions.detect_collisions_in_junction(0.5)
                    or gap <= 0.3
                    or truncated
                ):
                    first_collision_detected = True

                    current_position = traci.vehicle.getPosition(vehicle_id)
                    dx = current_position[0] - init_position[0]
                    dy = current_position[1] - init_position[1]

                    first_collision_distance = (dx**2 + dy**2)**0.5
                    first_collision_step = step_count

                # Compute impact on follower vehicles
                num_affected, avg_decel = env.get_followers_deceleration(vehicle_id)

            else:
                pass

            terminal = done or truncated or (not vehicle_exists)

            state = next_state
            total_reward += reward
            step_count += 1

            if terminal:
                break

        # Compute final metrics for the episode
        avg_speed = np.mean(episode_speeds) if episode_speeds else 0

        # The episode is successful only if there was no collision,
        # the reward was not extremely negative, and the vehicle actually moved.
        success = (
            not first_collision_detected
            and total_reward > -1000
            and avg_speed > 0.1
        )

        # Save episode metrics
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

        print(f"Validation episode {episode} - Total reward: {total_reward:.2f}, Success: {success}")

    # Close SUMO only once after validation
    try:
        traci.close()
    except:
        pass

    accuracy = metrics.success_per_episode

    if len(accuracy) > 0:
        print(f"Accuracy: {sum(accuracy) * 100 / len(accuracy):.2f}%")
    else:
        print("Accuracy: 0.00%")

    # Save validation metrics
    metrics.save(agent_id, save_metric)

    # Plot validation metrics
    metrics.plot(agent_id, episodes)

    print(f"PPO validation completed for {agent_id}.")