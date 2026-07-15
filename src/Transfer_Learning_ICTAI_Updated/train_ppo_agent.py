from ppo import PPOAgent
from sumo_gym_env_ppo import SumoGymEnvPPO
import numpy as np
import traci
from metrics import Metrics
from Collisions import Collisions_veh


def train_ppo(agent_id, sumo_config, vehicle_id, save_model, save_metric,
              edges_list, color, episodes=1000, sumo_gui=False):

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


    # Create SUMO environment for PPO
    env = SumoGymEnvPPO(sumo_cmd, vehicle_id, edges_list, color)

    # Initialize PPO agent
    agent = PPOAgent(
        state_size=env.observation_space.n,
        action_size=env.action_space.n,
        lr=3e-4,
        gamma=0.99,
        clip_epsilon=0.2,
        ppo_epochs=4,
        value_coef=0.1,
        entropy_coef=0.02,
        debug=False
    )

    metrics = Metrics()

    # Training loop
    for episode in range(episodes):
        print(f"Agent {agent_id} - Épisode {episode}")

        # Reset environment at the beginning of each episode.
        # SUMO is not closed here anymore.
        # The environment reset should reload the scenario using traci.load(),
        # so the simulation starts fresh without reopening the SUMO window every time.
        state, _ = env.reset()


        # Sometimes the controlled vehicle may not be loaded correctly after reset
        # In that case, I skip the episode instead of continuing with an invalid vehicle
        if vehicle_id not in traci.vehicle.getIDList():
            print(f"Episode {episode}: vehicle {vehicle_id} not found after reset.")
            continue

        init_position = traci.vehicle.getPosition(vehicle_id)

        traci.vehicle.setSpeedMode(vehicle_id, 0)

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

        vehicle_exists = True

        while not done and not truncated:
            if vehicle_id not in traci.vehicle.getIDList():
                vehicle_exists = False
                break

            # Select action using PPO policy
            # PPO also returns log_prob and value, because they are needed
            # later to compare the old policy with the updated policy.
            action, log_prob, value = agent.get_action(state)

            # Apply the selected action in the SUMO and receieve the next state and reward
            # This is the interaction step between the PPO agent and the traffic environment.
            next_state, reward, done, truncated, _ = env.step(action)

            vehicle_exists = vehicle_id in traci.vehicle.getIDList()

            if vehicle_exists:
                current_speed = traci.vehicle.getSpeed(vehicle_id)
                episode_speeds.append(current_speed)

                leader_info = traci.vehicle.getLeader(vehicle_id)
                gap = leader_info[1] if leader_info else float('inf')

                # Check if this is the first collision or dangerous situation in the episode.
                # I only save the first one because the metrics need the first collision time
                # and distance, not repeated collision detections.
                if not first_collision_detected and (
                    vehicle_id in collisions.detect_collisions()
                    or vehicle_id in collisions.detect_collisions_in_junction(0.5)
                    or gap < 0
                    or truncated
                ):
                    first_collision_detected = True

                    current_position = traci.vehicle.getPosition(vehicle_id)
                    dx = current_position[0] - init_position[0]
                    dy = current_position[1] - init_position[1]

                    first_collision_distance = (dx**2 + dy**2)**0.5
                    first_collision_step = step_count

                # Measure how much the controlled vehicle affects the vehicles behind it.
                # This helps to see if the agent causes urgency braking for follower vehicles.
                num_affected, avg_decel = env.get_followers_deceleration(vehicle_id)


            # The episode is finished if the goal is reached, the episode is truncated,
            # or the controlled vehicle has left the simulation.
            terminal = done or truncated or (not vehicle_exists)

            # Store this step in PPO memory
            # This memory is temporary and contains the trajectory of the current episode.
            agent.store(state, action, reward, terminal, log_prob, value)

            state = next_state
            total_reward += reward
            step_count += 1

            if terminal:
                break

        # Update the PPO policy after collecting one complete episode.
        # After this update, the trajectory data has already been used,
        # so the memory is cleared inside agent.update().
        agent.update()

        # In training, I count the episode as successful if no collision
        # or dangerous situation was detected.
        success = not first_collision_detected
        avg_speed = np.mean(episode_speeds) if episode_speeds else 0

        # Save episode metrics
        metrics.update(
            total_reward, step_count, first_collision_step,
            first_collision_distance, success, avg_speed,
            num_affected, avg_decel
        )

        print(f"Épisode {episode} - Total reward: {total_reward:.2f}")

        # Save PPO checkpoint every 50 episodes
        if (episode + 1) % 50 == 0:
            checkpoint_path = save_model.replace(".pth", f"_ep{episode+1}.pth")
            agent.save(checkpoint_path)
            print(f"PPO checkpoint saved: {checkpoint_path}")

    # Close SUMO only once after all training episodes are finished.
    traci.close()

    # Save the final trained PPO model after all episodes are completed.
    agent.save(save_model)
    print(f"PPO model saved: {save_model}")

    # Save and plot the training metrics for later comparison.
    metrics.save(agent_id, save_metric)
    metrics.plot(agent_id, episodes)