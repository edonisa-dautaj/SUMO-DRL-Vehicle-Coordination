# Implementation of Reinforcement Learning and Deep Reinforcement Learning Algorithms in Multi-Agent Systems for Autonomous Vehicle Coordination

This repository contains the code and experimental files for my research project at INSA Rouen Normandie.

The project focuses on reinforcement learning and deep reinforcement learning methods for autonomous vehicle coordination in SUMO traffic simulations. The work is based on an existing framework that already included Q-Learning and DQN. During this project, I added and tested DDQN and PPO implementations.

## Project Overview

The experiments are carried out using SUMO and TraCI. The controlled vehicle learns to choose discrete actions such as accelerating, maintaining speed, or decelerating.

Two main scenarios are used:

* Red agent in an intersection scenario
* Orange agent in a ramp merge scenario

## Implemented Algorithms

The project includes:

* Q-Learning
* DQN
* DDQN
* PPO

Q-Learning and DQN were part of the original framework. DDQN and PPO were added during this work.

## Repository Content

The repository includes:

* SUMO configuration files
* Training and validation scripts
* DDQN and PPO implementation files
* Saved model files
* Training and validation metrics
* Figures used in the report

## Current Status

DDQN was integrated into the existing DQN-based pipeline and produced stable validation results.

PPO was also implemented as an actor-critic model. The Red PPO agent showed promising behavior in the intersection scenario, while the Orange PPO agent in the ramp merge scenario still needs further improvement. Therefore, the PPO results should be considered preliminary.

## Future Work

Possible future improvements include:

* Improving the PPO state representation
* Adjusting the reward function for safer merging behavior
* Tuning PPO hyperparameters
* Testing under different traffic densities
* Extending the work to transfer learning and policy fusion experiments

## Author

Edonisa Dautaj

Supervised by Dr. Maxime Guériau and Fatima-Ezzahra Maad
INSA Rouen Normandie, LITIS UR 4108, France
