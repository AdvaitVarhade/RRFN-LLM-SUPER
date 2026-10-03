import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from robust_super.src.data.amazon_loader import AmazonReviewLoader

def test_amazon_loader_synthetic_fallback():
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = AmazonReviewLoader(data_dir=tmpdir, min_user_interactions=2, min_item_interactions=2)
        ratings_df, items_df, users_df = loader.load_data()

        assert len(ratings_df) > 0
        assert "user_id" in ratings_df.columns
        assert "item_id" in ratings_df.columns
        assert "rating" in ratings_df.columns
        assert "timestamp" in ratings_df.columns

        # Verify contiguous 0-indexed mappings
        assert ratings_df["user_id"].min() == 0
        assert ratings_df["user_id"].max() == loader.num_users - 1
        assert ratings_df["item_id"].min() == 0
        assert ratings_df["item_id"].max() == loader.num_items - 1

def test_amazon_loader_kcore_convergence():
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = AmazonReviewLoader(data_dir=tmpdir, min_user_interactions=3, min_item_interactions=3)
        ratings_df, items_df, users_df = loader.load_data()

        user_counts = ratings_df["user_id"].value_counts()
        item_counts = ratings_df["item_id"].value_counts()

        assert (user_counts >= 3).all()
        assert (item_counts >= 3).all()
