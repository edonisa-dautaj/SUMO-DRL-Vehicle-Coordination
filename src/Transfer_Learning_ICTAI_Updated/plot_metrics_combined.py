import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_multiple_csvs(csv_paths, agent_ids=None, save_dir="./metrics/combined_metrics_expert_agents"):
    if agent_ids is None:
        agent_ids = [f"Agent {i+1}" for i in range(len(csv_paths))]

    os.makedirs(save_dir, exist_ok=True)

    # Liste des métriques à tracer (doit correspondre aux colonnes CSV)
    metrics_to_plot = {
        "Average_Reward": "Mean Reward",
        "Travel_Time": "Travel Time (s)",
        "Time_to_Collision": "Time to Collision (s)",
        "Distance_to_Collision": "Distance to Collision (m)",
        "Success": "Success (1=Yes, 0=No)",
        "Average_Speed": "Mean Speed (m/s)",
        "Affected_Followers": "Slowed Vehicles",
        "Avg_Deceleration": "Mean Deceleration (m/s²)"
    }

    for metric, label in metrics_to_plot.items():
        plt.figure(figsize=(12, 6))
        for csv_path, agent_id in zip(csv_paths, agent_ids):
            df = pd.read_csv(csv_path)
            y = df[metric].fillna(0)
            x = df["Episode"]
            plt.plot(x, y, label=agent_id)

        plt.title(label, fontsize=24)
        plt.xlabel("Episodes", fontsize=24)
        plt.ylabel(label, fontsize=24)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.grid(True)
        plt.legend(fontsize=16)
        plt.tight_layout()

        filename = f"{metric}.pdf".replace(" ", "_")
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        
        plt.show()