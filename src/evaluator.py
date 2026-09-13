import numpy as np
import pandas as pd

def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error (Accuracy)"""
    return np.sqrt(np.mean((y_true - y_pred)**2))

def calculate_gini_index(item_exposure_counts):
    """
    Calculate Gini Index to measure inequality of item exposures.
    Values closer to 0 mean equal exposure (fairness).
    Values closer to 1 mean extreme popularity bias.
    """
    if len(item_exposure_counts) == 0:
        return 0.0
    
    # Sort counts ascending
    sorted_counts = np.sort(item_exposure_counts)
    n = len(sorted_counts)
    cumulative_sum = np.cumsum(sorted_counts)
    total_sum = cumulative_sum[-1]
    
    if total_sum == 0:
        return 0.0
        
    # Gini formula
    gini = (n + 1 - 2 * (np.sum(cumulative_sum) / total_sum)) / n
    return gini

def evaluate_novelty(recommendations_df, popularity_dict):
    """
    Calculate novelty based on how unpopular the recommended items are globally.
    popularity_dict maps item_id -> global probability of exposure.
    Novelty = -log2(p_i)
    """
    novelty_scores = []
    for item in recommendations_df['movie_id']:
        p = popularity_dict.get(item, 0.0001)  # small epsilon for unseen items
        novelty_scores.append(-np.log2(p))
    return np.mean(novelty_scores)

if __name__ == "__main__":
    # Dummy tests
    true_ratings = np.array([4.0, 5.0, 2.0, 1.0])
    pred_ratings = np.array([3.8, 4.9, 2.5, 1.2])
    print(f"RMSE: {calculate_rmse(true_ratings, pred_ratings):.4f}")
    
    # Gini test: completely equal exposure
    equal = np.array([10, 10, 10, 10])
    print(f"Gini (Equal): {calculate_gini_index(equal):.4f}")
    
    # Gini test: extreme bias
    biased = np.array([1000, 10, 2, 0])
    print(f"Gini (Biased): {calculate_gini_index(biased):.4f}")
