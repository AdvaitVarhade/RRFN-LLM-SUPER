import os
import json
import gzip
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List

class AmazonReviewLoader:
    """
    Parser and loader for Amazon Product Reviews datasets (e.g., Electronics_5, Books_5).
    Supports loading from .json or .json.gz files, and falls back to synthetic data if missing.
    """
    def __init__(
        self,
        data_dir: str,
        category: str = "Electronics",
        min_user_interactions: int = 5,
        min_item_interactions: int = 5
    ):
        self.data_dir = data_dir
        self.category = category
        self.min_user_interactions = min_user_interactions
        self.min_item_interactions = min_item_interactions
        
        self.user_to_idx: Dict[str, int] = {}
        self.idx_to_user: Dict[int, str] = {}
        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_item: Dict[int, str] = {}
        
        self.num_users: int = 0
        self.num_items: int = 0
        self.global_mean_rating: float = 4.0

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads Amazon reviews. Checks for {category}_5.json.gz, {category}_5.json, or fallback synthetic.
        Returns (ratings_df, items_df, users_df).
        """
        gz_path = os.path.join(self.data_dir, f"{self.category}_5.json.gz")
        json_path = os.path.join(self.data_dir, f"{self.category}_5.json")
        alt_gz = os.path.join(self.data_dir, "amazon_reviews.json.gz")
        alt_json = os.path.join(self.data_dir, "amazon_reviews.json")

        target_file = None
        for p in [gz_path, json_path, alt_gz, alt_json]:
            if os.path.exists(p):
                target_file = p
                break

        if target_file is None:
            return self._generate_synthetic_data()

        rows = []
        is_gz = target_file.endswith(".gz")
        opener = gzip.open(target_file, "rt", encoding="utf-8") if is_gz else open(target_file, "r", encoding="utf-8")
        
        try:
            for line in opener:
                if not line.strip():
                    continue
                item = json.loads(line)
                reviewer_id = item.get("reviewerID")
                asin = item.get("asin")
                overall = item.get("overall")
                time_val = item.get("unixReviewTime")
                review_text = item.get("reviewText", "")
                
                if reviewer_id and asin and overall is not None:
                    rows.append({
                        "user_id": str(reviewer_id),
                        "item_id": str(asin),
                        "rating": int(round(float(overall))),
                        "timestamp": int(time_val) if time_val is not None else 1000000,
                        "review_text": str(review_text)
                    })
        finally:
            opener.close()

        if not rows:
            return self._generate_synthetic_data()

        df = pd.DataFrame(rows)
        df = self._filter_k_core(df)

        # Build contiguous 0-indexed mappings
        unique_users = sorted(df["user_id"].unique())
        unique_items = sorted(df["item_id"].unique())

        self.user_to_idx = {u: idx for idx, u in enumerate(unique_users)}
        self.idx_to_user = {idx: u for idx, u in enumerate(unique_users)}
        self.item_to_idx = {i: idx for idx, i in enumerate(unique_items)}
        self.idx_to_item = {idx: i for idx, i in enumerate(unique_items)}

        self.num_users = len(unique_users)
        self.num_items = len(unique_items)

        df["user_id"] = df["user_id"].map(self.user_to_idx)
        df["item_id"] = df["item_id"].map(self.item_to_idx)
        df["rating"] = df["rating"].astype(int).clip(1, 5)
        ratings_df = df.sort_values(by=["user_id", "timestamp"]).reset_index(drop=True)

        items_df = pd.DataFrame({
            "item_id": np.arange(self.num_items),
            "title": [f"Amazon_{self.category}_{self.idx_to_item[i]}" for i in range(self.num_items)],
            "genres": [[self.category] for _ in range(self.num_items)]
        })

        users_df = pd.DataFrame({
            "user_id": np.arange(self.num_users),
            "gender": ["U"] * self.num_users,
            "age": [30] * self.num_users,
            "occupation": [0] * self.num_users,
            "zip": ["00000"] * self.num_users
        })

        self.global_mean_rating = float(ratings_df["rating"].mean())
        return ratings_df, items_df, users_df

    def _filter_k_core(self, df: pd.DataFrame) -> pd.DataFrame:
        """Iteratively filters users and items with insufficient interactions until convergence."""
        while True:
            u_counts = df["user_id"].value_counts()
            valid_users = u_counts[u_counts >= self.min_user_interactions].index
            
            i_counts = df["item_id"].value_counts()
            valid_items = i_counts[i_counts >= self.min_item_interactions].index

            new_df = df[df["user_id"].isin(valid_users) & df["item_id"].isin(valid_items)]
            if len(new_df) == len(df):
                break
            df = new_df
        return df

    def _generate_synthetic_data(
        self,
        num_users: int = 60,
        num_items: int = 120,
        num_interactions: int = 1500
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generates synthetic micro dataset representing Amazon product reviews."""
        np.random.seed(42)
        users = [f"AZ_USER_{u:04d}" for u in np.random.randint(0, num_users, size=num_interactions)]
        
        # Power law distribution for products
        item_probs = 1.0 / (np.arange(1, num_items + 1) ** 0.85)
        item_probs /= item_probs.sum()
        raw_items = np.random.choice(np.arange(num_items), size=num_interactions, p=item_probs)
        items = [f"B00{i:05d}X" for i in raw_items]
        
        ratings = np.random.choice([1, 2, 3, 4, 5], size=num_interactions, p=[0.08, 0.08, 0.15, 0.30, 0.39])
        timestamps = np.sort(np.random.randint(1300000000, 1600000000, size=num_interactions))
        
        raw_df = pd.DataFrame({
            "user_id": users,
            "item_id": items,
            "rating": ratings,
            "timestamp": timestamps,
            "review_text": ["Great product, fits perfectly." if r >= 4 else "Defective and slow delivery." for r in ratings]
        }).drop_duplicates(subset=["user_id", "item_id"]).reset_index(drop=True)

        df = self._filter_k_core(raw_df)

        unique_users = sorted(df["user_id"].unique())
        unique_items = sorted(df["item_id"].unique())

        self.user_to_idx = {u: idx for idx, u in enumerate(unique_users)}
        self.idx_to_user = {idx: u for idx, u in enumerate(unique_users)}
        self.item_to_idx = {i: idx for idx, i in enumerate(unique_items)}
        self.idx_to_item = {idx: i for idx, i in enumerate(unique_items)}

        self.num_users = len(unique_users)
        self.num_items = len(unique_items)

        df["user_id"] = df["user_id"].map(self.user_to_idx)
        df["item_id"] = df["item_id"].map(self.item_to_idx)
        df["rating"] = df["rating"].astype(int)
        ratings_df = df.sort_values(by=["user_id", "timestamp"]).reset_index(drop=True)

        items_df = pd.DataFrame({
            "item_id": np.arange(self.num_items),
            "title": [f"Amazon Product {self.idx_to_item[i]}" for i in range(self.num_items)],
            "genres": [[self.category] for _ in range(self.num_items)]
        })

        users_df = pd.DataFrame({
            "user_id": np.arange(self.num_users),
            "gender": ["U"] * self.num_users,
            "age": [30] * self.num_users,
            "occupation": [0] * self.num_users,
            "zip": ["00000"] * self.num_users
        })

        self.global_mean_rating = float(ratings_df["rating"].mean())
        return ratings_df, items_df, users_df
