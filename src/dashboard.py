import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os

def plot_pareto_front(rmse_scores, fairness_scores, save_path="pareto_front.png"):
    """
    Plot the trade-off between RMSE and Fairness Loss.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=rmse_scores, y=fairness_scores, s=100, color='b')
    plt.title("Pareto Front: Trade-off between RMSE and Fairness Loss")
    plt.xlabel("RMSE (Lower is Better)")
    plt.ylabel("Fairness Loss / Popularity Bias (Lower is Better)")
    plt.grid(True)
    
    # Annotate points
    for i, (x, y) in enumerate(zip(rmse_scores, fairness_scores)):
        plt.annotate(f"Sol {i+1}", (x, y), textcoords="offset points", xytext=(0,10), ha='center')
        
    plt.savefig(save_path)
    plt.close()
    print(f"Pareto front plot saved to {save_path}")

def plot_exposure_distribution(item_counts_base, item_counts_dp, save_path="exposure.png"):
    """
    Compare item exposure between Base model and DP-Fair model.
    """
    plt.figure(figsize=(12, 6))
    
    # Sort items by base exposure
    indices = np.argsort(item_counts_base)[::-1]
    sorted_base = np.array(item_counts_base)[indices]
    sorted_dp = np.array(item_counts_dp)[indices]
    
    x = np.arange(len(sorted_base))
    
    plt.plot(x, sorted_base, label="Base Model (Biased)", color='red', alpha=0.7)
    plt.plot(x, sorted_dp, label="DP-Fair Model", color='green', alpha=0.7)
    
    plt.fill_between(x, sorted_base, alpha=0.3, color='red')
    plt.fill_between(x, sorted_dp, alpha=0.3, color='green')
    
    plt.title("Long-Tail Item Exposure Comparison")
    plt.xlabel("Item Rank (Popular to Unpopular)")
    plt.ylabel("Exposure Count")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.savefig(save_path)
    plt.close()
    print(f"Exposure plot saved to {save_path}")

if __name__ == "__main__":
    # Dummy data for testing
    sim_rmse = [0.82, 0.85, 0.88, 0.95, 1.05]
    sim_fairness = [0.45, 0.40, 0.35, 0.28, 0.20]
    
    plot_pareto_front(sim_rmse, sim_fairness)
    
    base_counts = np.geomspace(1000, 1, num=100)
    dp_counts = np.geomspace(500, 5, num=100)
    plot_exposure_distribution(base_counts, dp_counts)
    
    # Export data for web dashboard
    dashboard_data = {
        "pareto": {
            "rmse": sim_rmse,
            "fairness": sim_fairness
        },
        "exposure": {
            "base": base_counts.tolist(),
            "dp": dp_counts.tolist()
        }
    }
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dashboard-ui")
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "dashboard_data.json")
    with open(json_path, 'w') as f:
        json.dump(dashboard_data, f)
    print(f"Data exported for web dashboard at {json_path}")
