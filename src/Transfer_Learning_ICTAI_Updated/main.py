from plot_metrics_combined import plot_multiple_csvs
from train_agent import train_agent
from train_domain_adaptation import train_with_transfer
from validation_agent import validate_agent
from validation_agent_dqn import validate_dqn_agent
from validation_agent_ddqn import validate_ddqn_agent
from train_dqn_agent import train_dqn
from train_ddqn_agent import train_ddqn
from train_ppo_agent import train_ppo
from validation_agent_ppo import validate_ppo_agent
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():

    # Configuration des agents à entraîner

    agents_config = [
        # Red from scratch 1000 episodes
        {"agent_id": "Red", "sumo_config": "./configurations/Multi_agent/red_agent.sumocfg", "vehicle_id": "v_red", "save_file": "./models/ICTAI/q_table_agent_red_ICTAI_2.npy", "save_metric": "./metrics/Red_agent_scratch_intersection/training/training_metrics_agent_v_red_2.csv", "edges_list": ('-E0', 'E1'), "color": (255, 0, 0, 255), "sumo_gui": True},

        # Orange from scratch 1000 episodes
        #{"agent_id": "Orange","sumo_config": "./configurations/Multi_agent/orange_agent.sumocfg", "vehicle_id": "v_orange", "save_file": "./models/ICTAI/q_table_agent_orange_ICTAI_2.npy", "save_metric" : "./metrics/Orange_agent_scratch_merge/training/training_metrics_agent_v_orange_2.csv", "edges_list": ('E1', 'E0.58'), "color": (255, 165, 0, 255),"sumo_gui": False},

        # Red dans l'env merge (pretrained) without help
        #{"agent_id": "Red", "sumo_config": "./configurations/Multi_agent/orange_agent.sumocfg", "vehicle_id": "v_orange", "save_file": "./models/ICTAI/q_table_agent_red_pretrained_intersection_then_merge_2.npy", "save_metric" : "./metrics/Pretrained_Red_agent_whitout_orange_help/training/training_metrics_pretrained_agent_v_red_2.csv", "edges_list": ('E1', 'E0.58'), "color": (255, 0, 0, 255), "episodes" : 100, "sumo_gui": True},

        # Orange dans l'env insertion (pretrained) without help
        #{"agent_id": "Orange", "sumo_config": "./configurations/Multi_agent/red_agent.sumocfg", "vehicle_id": "v_red", "save_file": "./models/ICTAI/q_table_agent_orange_pretrained_merge_then_intersection_2.npy", "save_metric" : "./metrics/Pretrained_Orange_agent_whitout_red_help/training/training_metrics_pretrained_agent_v_orange_2.csv", "edges_list": ('-E0', 'E1'), "color": (255, 165, 0, 255), "episodes" : 100, "sumo_gui": True},
    ]

    # Configuration des agents à valider

    agents_validation = [
        # Red from scratch
        {"agent_id" : "red", "sumo_config" : "./configurations/Multi_agent/red_agent.sumocfg", "vehicle_id" : "v_red", "q_table_file" : "./models/ICTAI/q_table_agent_red_ICTAI_2.npy", "save_metric": "./metrics/Red_agent_scratch_intersection/validation/validation_metrics_agent_v_red_2.csv", "edges_list" : ('-E0', 'E1'), "color" : (255, 0, 0), "episodes" : 100, "sumo_gui" : True},

        # Orange from scratch
        #{"agent_id" : "orange", "sumo_config" : "./configurations/Multi_agent/orange_agent.sumocfg", "vehicle_id" : "v_orange", "q_table_file" : "./models/ICTAI/q_table_agent_orange_ICTAI_2.npy", "save_metric" : "./metrics/Orange_agent_scratch_merge/validation/validation_metrics_agent_v_orange_2.csv", "edges_list" : ('E1', 'E0.58'), "color" : (255, 165, 0), "episodes" : 100, "sumo_gui" : False},

        # Red dans l'env merge (pretrained) without help
        {"agent_id": "Red", "sumo_config": "./configurations/Multi_agent/orange_agent.sumocfg", "vehicle_id": "v_orange","q_table_file": "./models/ICTAI/q_table_agent_red_pretrained_intersection_then_merge_2.npy", "save_metric": "./metrics/Pretrained_Red_agent_whitout_orange_help/validation/validation_metrics_pretrained_agent_v_red_2.csv", "edges_list" : ('E1', 'E0.58'), "color" : (255, 0, 0), "episodes" : 100, "sumo_gui": True},

        # Orange dans l'env insertion (pretrained) without help
        #{"agent_id" : "Orange", "sumo_config" : "./configurations/Multi_agent/red_agent.sumocfg", "vehicle_id" : "v_red", "q_table_file" : "./models/ICTAI/q_table_agent_orange_pretrained_merge_then_intersection_2.npy", "save_metric" : "./metrics/Pretrained_Orange_agent_whitout_red_help/validation/validation_metrics_pretrained_agent_v_orange_2.csv","edges_list": ('-E0', 'E1'), "color": (255, 165, 0, 255), "episodes" : 100, "sumo_gui": True},

        # Red agent zero-shot model 1100 episode validation
        #{"agent_id": "red", "sumo_config": "./configurations/Multi_agent/orange_agent.sumocfg", "vehicle_id": "v_orange", "q_table_file": "./models/ICTAI/q_table_agent_red_ICTAI_2.npy", "save_metric" : "./metrics/zero_shot_red_agent_validation/validation_metrics_agent_v_red_2.csv", "edges_list": ('E1', 'E0.58'), "color": (255, 0, 0), "episodes": 100, "sumo_gui": True},

        # Orange agent zero-shot model 1100 episode validation
        #{"agent_id": "orange", "sumo_config": "./configurations/Multi_agent/red_agent.sumocfg", "vehicle_id": "v_red", "q_table_file": "./models/ICTAI/q_table_agent_orange_ICTAI_2.npy", "save_metric": "./metrics/zero_shot_orange_agent_validation/validation_metrics_agent_v_orange_2.csv", "edges_list": ('-E0', 'E1'), "color": (255, 165, 0, 255), "episodes": 100, "sumo_gui": True},

    ]

    transfer_finetuning = [
    # Red dans l'env merge (pretrained) with help TF + FINETUNE
    #{"agent_id" : "Red", "sumo_config" : "./configurations/Multi_agent/orange_agent.sumocfg", "vehicle_id" :  "v_orange", "own_q_file" : "./models/ICTAI/q_table_agent_red_ICTAI_2.npy", "other_q_file" : "./models/ICTAI/q_table_agent_orange_ICTAI_2.npy", "save_file" : "./models/ICTAI/Red_agent_in_merge_with_orange_help_2.npy", "save_metric": "./metrics/Red_agent_in_merge_with_orange_help/finetuning_metrics_agent_Red_with_orange_help_2.csv", "edges_list": ('E1', 'E0.58'), "color": (255, 0, 0, 255), "episodes" : 100,"sumo_gui" : True},

    # Orange dans l'env intersection (pretrained) with help TF + FINETUNE
    #{"agent_id": "Orange", "sumo_config": "./configurations/Multi_agent/red_agent.sumocfg", "vehicle_id": "v_red", "own_q_file": "./models/ICTAI/q_table_agent_orange_ICTAI_2.npy", "other_q_file" : "./models/ICTAI/q_table_agent_red_ICTAI_2.npy" , "save_file": "./models/ICTAI/Orange_agent_in_intersection_with_red_help_2.npy", "save_metric": "./metrics/Orange_agent_in_intersection_with_red_help/finetuning_metrics_agent_Orange_with_red_help_2.csv", "edges_list": ('-E0', 'E1'), "color": (255, 165, 0, 255), "episodes": 100, "sumo_gui": True}
    ]


    #for config in agents_config:
        #train_agent(**config)


    #for val in agents_validation:
        #validate_agent(**val)


    """
    for tl in transfer_finetuning:
        train_with_transfer(**tl)
    """


if __name__ == "__main__":
    #main()
    #train Red agent expert dqn
    #train_dqn("Red", "./configurations/Multi_agent/red_agent.sumocfg", "v_red", "./models/ICTAI/DQN_agent_red_ICTAI_2.pth", "./metrics/Red_agent_scratch_intersection/training/DQN_training_metrics_agent_v_red_2.csv", ('-E0', 'E1'), (255, 0, 0, 255), 1000, False)

    #validate Red agent expert dqn
    #validate_dqn_agent("Red", "./configurations/Multi_agent/red_agent.sumocfg", "v_red", "./models/ICTAI/DQN_agent_red_ICTAI_2.pth","./metrics/Red_agent_scratch_intersection/validation/DQN_validation_metrics_agent_v_red_2.csv", ('-E0', 'E1'), (255, 0, 0, 255), 1000, True)

    #zero shot red dqn
    #validate_dqn_agent("Red", "./configurations/Multi_agent/orange_agent.sumocfg", "v_orange", "./models/ICTAI/DQN_agent_red_ICTAI_2.pth","./metrics/zero_shot_red_agent_validation/dqn_validation_metrics_pretrained_agent_v_red_2.csv", ('E1', 'E0.58'),  (255, 0, 0), 100, True)

    # train orange agent expert dqn
    #train_dqn("Orange", "./configurations/Multi_agent/orange_agent.sumocfg", "v_orange", "./models/ICTAI/DQN_agent_orange_ICTAI_2.pth", "./metrics/Orange_agent_scratch_merge/training/DQN_training_metrics_agent_v_orange_2.csv", ('E1', 'E0.58'), (255, 165, 0, 255), 1000, False)

    # validate Orange agent expert dqn
    #validate_dqn_agent("Orange", "./configurations/Multi_agent/orange_agent.sumocfg", "v_orange", "./models/ICTAI/DQN_agent_orange_ICTAI_2.pth","./metrics/Orange_agent_scratch_merge/validation/DQN_validation_metrics_agent_v_orange_2.csv", ('E1', 'E0.58'), (255, 165, 0, 255), 100, True)

    # zero shot orange dqn
    #validate_dqn_agent("Orange", "./configurations/Multi_agent/red_agent.sumocfg", "v_red", "./models/ICTAI/DQN_agent_orange_ICTAI_2.pth","./metrics/zero_shot_orange_agent_validation/dqn_validation_metrics_pretrained_agent_v_orange_2.csv", ('-E0', 'E1'),  (255, 165, 0, 255), 100, True)

    # train Red agent expert ddqn
    #train_ddqn("Red", "./configurations/Multi_agent/red_agent.sumocfg", "v_red","./models/ICTAI/DDQN_agent_red_ICTAI_2.pth","./metrics/Red_agent_scratch_intersection/training/DDQN_training_metrics_agent_v_red_2.csv", ('-E0', 'E1'), (255, 0, 0, 255), 100, True)

    # validate Red agent ddqn
    #validate_ddqn_agent("Red", "./configurations/Multi_agent/red_agent.sumocfg", "v_red","./models/ICTAI/DDQN_agent_red_ICTAI_2.pth","./metrics/Red_agent_scratch_intersection/validation/DDQN_validation_metrics_agent_v_red_2.csv",('-E0', 'E1'), (255, 0, 0, 255), 1000, True)

    # train Orange agent expert ddqn
    #train_ddqn("Orange","./configurations/Multi_agent/orange_agent.sumocfg","v_orange","./models/ICTAI/DDQN_agent_orange_ICTAI_2.pth","./metrics/Orange_agent_scratch_merge/training/DDQN_training_metrics_agent_v_orange_2.csv",('E1', 'E0.58'),(255, 165, 0, 255),1000,False)

    # validate Orange agent ddqn
    #validate_ddqn_agent("Orange","./configurations/Multi_agent/orange_agent.sumocfg","v_orange","./models/ICTAI/DDQN_agent_orange_ICTAI_2.pth","./metrics/Orange_agent_scratch_merge/validation/DDQN_validation_metrics_agent_v_orange_2.csv",('E1', 'E0.58'),(255, 165, 0, 255),1000,True)

    # train Red agent expert ppo
    #train_ppo("Red", "./configurations/Multi_agent/red_agent.sumocfg", "v_red","./models/ICTAI/PPO_agent_red_ICTAI_2.pth","./metrics/Red_agent_scratch_intersection/training/PPO_training_metrics_agent_v_red_2.csv",('-E0', 'E1'), (255, 0, 0, 255), 1000, False)

    # validate Red agent ppo
    #validate_ppo_agent("Red", "./configurations/Multi_agent/red_agent.sumocfg", "v_red","./models/ICTAI/PPO_agent_red_ICTAI_2.pth","./metrics/Red_agent_scratch_intersection/validation/PPO_validation_metrics_agent_v_red_2.csv",('-E0', 'E1'), (255, 0, 0, 255), 500, True)

    # train Orange agent expert ppo
    #train_ppo("Orange", "./configurations/Multi_agent/orange_agent.sumocfg", "v_orange","./models/ICTAI/PPO_agent_orange_ICTAI_2.pth","./metrics/Orange_agent_scratch_merge/training/PPO_training_metrics_agent_v_orange_2.csv",('E1', 'E0.58'), (255, 165, 0, 255), 1000, False)

    # validate Orange agent ppo
    #validate_ppo_agent("Orange", "./configurations/Multi_agent/orange_agent.sumocfg", "v_orange","./models/ICTAI/PPO_agent_orange_ICTAI_2.pth","./metrics/Orange_agent_scratch_merge/validation/PPO_validation_metrics_agent_v_orange_2.csv",('E1', 'E0.58'), (255, 165, 0, 255), 500, True)

    """
    # plot Orange agent in intersection
    csv_files = [

        "./metrics/Red_agent_scratch_intersection/validation/validation_metrics_agent_v_red_mean.csv",  # agent expert
        "./metrics/zero_shot_orange_agent_validation/validation_metrics_agent_v_orange_mean.csv",  # zero-shot 1100 episode
        "./metrics/Red_agent_scratch_intersection/validation/DQN_validation_metrics_agent_v_red_mean.csv",# agent expertDQN
        "./metrics/zero_shot_orange_agent_validation/dqn_validation_metrics_agent_v_orange_mean.csv",# zero-shot 1000 +100 validation episode
        "./metrics/Pretrained_Orange_agent_whitout_red_help/validation/validation_metrics_pretrained_agent_v_orange_mean.csv", # TL 1000 + 100 episodes
        "./metrics/Orange_agent_in_intersection_with_red_help/finetuning_metrics_agent_Orange_with_red_help_mean.csv",# finetuning
    ]

    agent_names = ["QL","DQN", "ZS-QL", "ZS-DQN", "Basic TL", "TL + finetuning"]
    save_dir = "./metrics/Metrics_combined_Orange_in_intersection/Boxplot_mean_all"
    os.makedirs(save_dir, exist_ok=True)

    metric_columns = ["Average_Reward", "Time_to_Collision", "Travel_Time", "Distance_to_Collision", "Average_Speed", "Affected_Followers", "Avg_Deceleration"]

    """