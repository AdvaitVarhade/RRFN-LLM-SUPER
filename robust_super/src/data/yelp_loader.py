import os
import json
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List

class YelpLoader:
    """
    Parser and loader for the Yelp Open Dataset (academic review & business dataset).
    Supports loading from yelp_academic_dataset_review.json + business.json,
    and falls back to synthetic Yelp reviews if raw files are absent.
    """
    def __init__(
        self,
        data_dir: str,
        min_user_interactions: int = 5,
        min_item_interactions: int = 5
    ):
        self.data_dir = data_dir
        self.min_user_interactions = min_user_interactions
        self.min_item_interactions = min_item_interactions
        
        self.user_to_idx: Dict[str, int] = {}
        self.idx_to_user: Dict[int, str] = {}
        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_item: Dict[int, str] = {}
        
        self.num_users: int = 0
        self.num_items: int = 0
        self.global_mean_rating: float = 3.8

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads Yelp reviews and business metadata.
        Returns (ratings_df, businesses_df, users_df).
        """
        review_path = os.path.join(self.data_dir, "yelp_academic_dataset_review.json")
        business_path = os.path.join(self.data_dir, "yelp_academic_dataset_business.json")
        alt_review = os.path.join(self.data_dir, "yelp_reviews.json")

        target_review = None
        for p in [review_path, alt_review]:
            if os.path.exists(p):
                target_review = p
                break

        if target_review is None:
            return self._generate_synthetic_data()

        rows = []
        with open(target_review, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                u_id = item.get("user_id")
                b_id = item.get("business_id")
                stars = item.get("stars")
                date_str = item.get("date")
                text = item.get("text", "")
                
                if u_id and b_id and stars is not None:
                    # Convert ISO date to timestamp
                    try:
                        ts = int(pd.to_datetime(date_str).timestamp()) if date_str else 1000000
                    except Exception:
                        ts = 1000000

                    rows.append({
                        "user_id": str(u_id),
                        "item_id": str(b_id),
                        "rating": int(round(float(stars))),
                        "timestamp": ts,
                        "review_text": str(text)
                    })

        if not rows:
            return self._generate_synthetic_data()

        df = pd.DataFrame(rows)
        df = self._filter_k_core(df)

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

        # Parse categories from business file if available
        business_names = {}
        business_cats = {}
        if os.path.exists(business_path):
            with open(business_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    b = json.loads(line)
                    bid = b.get("business_id")
                    if bid in self.item_to_idx:
                        business_names[self.item_to_idx[bid]] = b.get("name", f"Business_{bid}")
                        cats = b.get("categories", "Local Business")
                        business_cats[self.item_to_idx[bid]] = [c.strip() for c in cats.split(",")] if cats else ["Local Business"]

        businesses_df = pd.DataFrame({
            "item_id": np.arange(self.num_items),
            "title": [business_names.get(i, f"Yelp Business {self.idx_to_item[i]}") for i in range(self.num_items)],
            "genres": [business_cats.get(i, ["Local Business", "Food"]) for i in range(self.num_items)]
        })

        users_df = pd.DataFrame({
            "user_id": np.arange(self.num_users),
            "gender": ["U"] * self.num_users,
            "age": [30] * self.num_users,
            "occupation": [0] * self.num_users,
            "zip": ["00000"] * self.num_users
        })

        self.global_mean_rating = float(ratings_df["rating"].mean())
        return ratings_df, businesses_df, users_df

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
        num_users: int = 70,
        num_items: int = 140,
        num_interactions: int = 1800
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generates synthetic micro dataset representing Yelp business reviews."""
        np.random.seed(42)
        users = [f"YELP_U_{u:04d}" for u in np.random.randint(0, num_users, size=num_interactions)]
        
        # Heavy-tail popularity for restaurants/businesses
        item_probs = 1.0 / (np.arange(1, num_items + 1) ** 0.82)
        item_probs /= item_probs.sum()
        raw_items = np.random.choice(np.arange(num_items), size=num_interactions, p=item_probs)
        items = [f"BIZ_{i:04d}_NY" for i in raw_items]
        
        ratings = np.random.choice([1, 2, 3, 4, 5], size=num_interactions, p=[0.10, 0.10, 0.18, 0.32, 0.30])
        timestamps = np.sort(np.random.randint(1400000000, 1650000000, size=num_interactions))
        
        raw_df = pd.DataFrame({
            "user_id": users,
            "item_id": items,
            "rating": ratings,
            "timestamp": timestamps,
            "review_text": ["Fantastic atmosphere and food!" if r >= 4 else "Horrible customer service." for r in ratings]
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

        categories_pool = ["Restaurants", "Nightlife", "Coffee & Tea", "Bars", "Shopping", "Beauty & Spas", "Bakeries"]
        businesses_df = pd.DataFrame({
            "item_id": np.arange(self.num_items),
            "title": [f"Yelp Venue {self.idx_to_item[i]}" for i in range(self.num_items)],
            "genres": [list(np.random.choice(categories_pool, size=np.random.randint(1, 3), replace=False)) for _ in range(self.num_items)]
        })

        users_df = pd.DataFrame({
            "user_id": np.arange(self.num_users),
            "gender": ["U"] * self.num_users,
            "age": [30] * self.num_users,
            "occupation": [0] * self.num_users,
            "zip": ["10001"] * self.num_users
        })

        self.global_mean_rating = float(ratings_df["rating"].mean())
        return ratings_df, businesses_df, users_df
