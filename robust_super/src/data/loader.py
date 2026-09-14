import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List

class MovieLensLoader:
    """
    Parser and loader for the MovieLens-1M dataset.
    Supports loading from standard .dat files or generating synthetic micro-data.
    """
    def __init__(self, data_dir: str, min_user_interactions: int = 5, min_item_interactions: int = 5):
        self.data_dir = data_dir
        self.min_user_interactions = min_user_interactions
        self.min_item_interactions = min_item_interactions
        
        self.user_to_idx: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}
        self.item_to_idx: Dict[int, int] = {}
        self.idx_to_item: Dict[int, int] = {}
        
        self.num_users: int = 0
        self.num_items: int = 0
        self.global_mean_rating: float = 3.5

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads ratings.dat, movies.dat, users.dat from data_dir.
        If files do not exist, falls back to research/data/ml-1m or synthetic generator.
        """
        ratings_path = os.path.join(self.data_dir, "ratings.dat")
        movies_path = os.path.join(self.data_dir, "movies.dat")
        users_path = os.path.join(self.data_dir, "users.dat")

        # Fallback paths
        if not os.path.exists(ratings_path):
            alt_path = os.path.join(os.path.dirname(__file__), "../../../research/data/ml-1m")
            if os.path.exists(os.path.join(alt_path, "ratings.dat")):
                ratings_path = os.path.join(alt_path, "ratings.dat")
                movies_path = os.path.join(alt_path, "movies.dat")
                users_path = os.path.join(alt_path, "users.dat")
            else:
                return self._generate_synthetic_data()

        # Parse ratings
        ratings_df = pd.read_csv(
            ratings_path,
            sep="::",
            engine="python",
            names=["user_id", "item_id", "rating", "timestamp"],
            encoding="latin-1"
        )

        # Parse movies
        movies_df = pd.read_csv(
            movies_path,
            sep="::",
            engine="python",
            names=["item_id", "title", "genres"],
            encoding="latin-1"
        )
        movies_df["genres"] = movies_df["genres"].apply(lambda x: x.split("|") if isinstance(x, str) else ["Unknown"])

        # Parse users
        if os.path.exists(users_path):
            users_df = pd.read_csv(
                users_path,
                sep="::",
                engine="python",
                names=["user_id", "gender", "age", "occupation", "zip"],
                encoding="latin-1"
            )
        else:
            unique_users = ratings_df["user_id"].unique()
            users_df = pd.DataFrame({"user_id": unique_users})

        # Iterative k-core filtering
        ratings_df = self._filter_k_core(ratings_df)

        # Build contiguous 0-indexed mappings
        unique_users = sorted(ratings_df["user_id"].unique())
        unique_items = sorted(ratings_df["item_id"].unique())

        self.user_to_idx = {u: idx for idx, u in enumerate(unique_users)}
        self.idx_to_user = {idx: u for idx, u in enumerate(unique_users)}
        self.item_to_idx = {i: idx for idx, i in enumerate(unique_items)}
        self.idx_to_item = {idx: i for idx, i in enumerate(unique_items)}

        self.num_users = len(unique_users)
        self.num_items = len(unique_items)

        # Remap IDs
        ratings_df["user_id"] = ratings_df["user_id"].map(self.user_to_idx)
        ratings_df["item_id"] = ratings_df["item_id"].map(self.item_to_idx)
        ratings_df["rating"] = ratings_df["rating"].astype(int)
        ratings_df = ratings_df.sort_values(by=["user_id", "timestamp"]).reset_index(drop=True)

        movies_df = movies_df[movies_df["item_id"].isin(self.item_to_idx.keys())].copy()
        movies_df["item_id"] = movies_df["item_id"].map(self.item_to_idx)
        movies_df = movies_df.sort_values(by="item_id").reset_index(drop=True)

        users_df = users_df[users_df["user_id"].isin(self.user_to_idx.keys())].copy()
        users_df["user_id"] = users_df["user_id"].map(self.user_to_idx)
        users_df = users_df.sort_values(by="user_id").reset_index(drop=True)

        self.global_mean_rating = float(ratings_df["rating"].mean())

        return ratings_df, movies_df, users_df

    def _filter_k_core(self, df: pd.DataFrame) -> pd.DataFrame:
        """Iteratively filters users and items with insufficient interactions."""
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

    def _generate_synthetic_data(self, num_users: int = 50, num_items: int = 100, num_interactions: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generates synthetic micro dataset for fast offline testing."""
        np.random.seed(42)
        users = np.random.randint(0, num_users, size=num_interactions)
        # Power-law item popularity
        item_probs = 1.0 / (np.arange(1, num_items + 1) ** 0.8)
        item_probs /= item_probs.sum()
        items = np.random.choice(np.arange(num_items), size=num_interactions, p=item_probs)
        ratings = np.random.choice([1, 2, 3, 4, 5], size=num_interactions, p=[0.05, 0.10, 0.25, 0.35, 0.25])
        timestamps = np.sort(np.random.randint(1000000, 2000000, size=num_interactions))

        ratings_df = pd.DataFrame({
            "user_id": users,
            "item_id": items,
            "rating": ratings,
            "timestamp": timestamps
        }).drop_duplicates(subset=["user_id", "item_id"]).reset_index(drop=True)

        genres_list = ["Action", "Comedy", "Drama", "Sci-Fi", "Thriller", "Horror", "Romance", "Adventure"]
        movies_df = pd.DataFrame({
            "item_id": np.arange(num_items),
            "title": [f"Movie_{i}" for i in range(num_items)],
            "genres": [list(np.random.choice(genres_list, size=np.random.randint(1, 3), replace=False)) for _ in range(num_items)]
        })

        users_df = pd.DataFrame({
            "user_id": np.arange(num_users),
            "gender": np.random.choice(["M", "F"], size=num_users),
            "age": np.random.choice([18, 25, 35, 45, 50], size=num_users),
            "occupation": np.random.randint(0, 20, size=num_users),
            "zip": [f"{np.random.randint(10000, 99999):05d}" for _ in range(num_users)]
        })

        self.num_users = num_users
        self.num_items = num_items
        self.global_mean_rating = float(ratings_df["rating"].mean())
        self.user_to_idx = {u: u for u in range(num_users)}
        self.idx_to_user = {u: u for u in range(num_users)}
        self.item_to_idx = {i: i for i in range(num_items)}
        self.idx_to_item = {i: i for i in range(num_items)}

        return ratings_df, movies_df, users_df
