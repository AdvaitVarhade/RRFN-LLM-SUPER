from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional
import numpy as np
import pandas as pd

@dataclass
class DataSplit:
    train_dict: Dict[int, List[Tuple[int, int, int]]]  # u -> [(item, rating, timestamp)]
    val_dict: Dict[int, Tuple[int, int, int]]          # u -> (item, rating, timestamp)
    test_dict: Dict[int, Tuple[int, int, int]]         # u -> (item, rating, timestamp)
    weight_dict: Dict[Tuple[int, int], float]          # (u, i) -> weight (init 1.0)
    item_counts: Dict[int, int]                         # i -> num train interactions
    user_mean_ratings: Dict[int, float]                # u -> avg rating in train
    num_users: int
    num_items: int
    items_df: pd.DataFrame
    users_df: pd.DataFrame
    ground_truth_labels: Dict[Tuple[int, int], int] = field(default_factory=dict) # (u, i) -> 1 genuine, 0 injected

def preprocess_dataset(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    users_df: pd.DataFrame,
    split_strategy: str = "leave_one_out"
) -> DataSplit:
    """
    Splits ratings into train, validation, and test sets.
    Under leave_one_out:
      - Last interaction per user -> Test
      - Second-to-last interaction -> Validation
      - Remaining interactions -> Train
    """
    train_dict: Dict[int, List[Tuple[int, int, int]]] = {}
    val_dict: Dict[int, Tuple[int, int, int]] = {}
    test_dict: Dict[int, Tuple[int, int, int]] = {}
    weight_dict: Dict[Tuple[int, int], float] = {}
    ground_truth_labels: Dict[Tuple[int, int], int] = {}
    item_counts: Dict[int, int] = {}
    user_mean_ratings: Dict[int, float] = {}

    # Strict bounds computation
    max_u = int(max(ratings_df["user_id"].max(), users_df["user_id"].max() if "user_id" in users_df else 0))
    max_i = int(max(ratings_df["item_id"].max(), movies_df["item_id"].max() if "item_id" in movies_df else 0))
    num_users = max_u + 1
    num_items = max_i + 1

    # Group ratings by user
    grouped = ratings_df.groupby("user_id")

    for u, group in grouped:
        sorted_records = group.sort_values(by="timestamp").to_dict("records")
        if len(sorted_records) < 3:
            continue

        test_record = sorted_records[-1]
        val_record = sorted_records[-2]
        train_records = sorted_records[:-2]

        test_dict[int(u)] = (int(test_record["item_id"]), int(test_record["rating"]), int(test_record["timestamp"]))
        val_dict[int(u)] = (int(val_record["item_id"]), int(val_record["rating"]), int(val_record["timestamp"]))

        train_list = []
        user_ratings_sum = 0
        for r in train_records:
            i = int(r["item_id"])
            rating = int(r["rating"])
            ts = int(r["timestamp"])
            train_list.append((i, rating, ts))
            weight_dict[(int(u), i)] = 1.0
            ground_truth_labels[(int(u), i)] = 1  # 1 = genuine
            item_counts[i] = item_counts.get(i, 0) + 1
            user_ratings_sum += rating

        train_dict[int(u)] = train_list
        user_mean_ratings[int(u)] = user_ratings_sum / max(1, len(train_list))

    return DataSplit(
        train_dict=train_dict,
        val_dict=val_dict,
        test_dict=test_dict,
        weight_dict=weight_dict,
        item_counts=item_counts,
        user_mean_ratings=user_mean_ratings,
        num_users=num_users,
        num_items=num_items,
        items_df=movies_df,
        users_df=users_df,
        ground_truth_labels=ground_truth_labels
    )
