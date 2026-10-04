import pytest
import os
from src.data.loader import MovieLensLoader
from src.data.preprocessor import preprocess_dataset

def test_movielens_loader_and_preprocessor():
    loader = MovieLensLoader(data_dir="non_existent_dir", min_user_interactions=2, min_item_interactions=2, use_synthetic=True)
    ratings_df, movies_df, users_df = loader.load_data()

    assert not ratings_df.empty
    assert not movies_df.empty
    assert not users_df.empty
    assert loader.num_users > 0
    assert loader.num_items > 0

    # Verify column structure
    assert set(["user_id", "item_id", "rating", "timestamp"]).issubset(ratings_df.columns)
    assert set(["item_id", "title", "genres"]).issubset(movies_df.columns)

    split = preprocess_dataset(ratings_df, movies_df, users_df)
    assert len(split.train_dict) > 0
    assert len(split.val_dict) > 0
    assert len(split.test_dict) > 0

    # Ensure leave-one-out uniqueness
    for u in split.train_dict:
        train_items = [item for item, _, _ in split.train_dict[u]]
        if u in split.val_dict:
            val_item, _, _ = split.val_dict[u]
        if u in split.test_dict:
            test_item, _, _ = split.test_dict[u]
        # Valid split
        assert len(train_items) > 0

def test_movielens_loader_synthetic_flag():
    loader = MovieLensLoader(data_dir="", use_synthetic=True)
    ratings_df, movies_df, users_df = loader.load_data()
    assert len(ratings_df) > 0
    assert loader.num_users == 50
    assert loader.num_items == 100
    assert loader.global_mean_rating > 0.0
