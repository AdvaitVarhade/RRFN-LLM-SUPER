"""
robust_super/tests/test_dataset_scaling.py
=========================================
Comprehensive empirical verification tests for Requirement R1:
Full-Scale Empirical Benchmarks & Dataset Scaling.

Verifies:
  1. Full MovieLens-1M dataset scaling (Nu=6,040, Ni=3,952, 1,000,209 interactions).
  2. Temporal leave-one-out partitioning with zero validation/test contamination.
  3. Venue-standard 5-core properties for Amazon Reviews (Electronics) (Nu>=1000, Ni>=1000, >=20K ratings, user & item min degree >= 5).
  4. Venue-standard 5-core properties for Yelp Academic (Nu>=1000, Ni>=1000, >=20K ratings, user & item min degree >= 5).
  5. Full-scale zero-leakage bandwagon attack injection (rho=0.15, N_shill = ceil(0.15 * 6040) = 906, strictly train_dict).
  6. Script data ingestion scale flags (full vs. sample) across all domains.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.data.loader import MovieLensLoader
from src.data.amazon_loader import AmazonReviewLoader
from src.data.yelp_loader import YelpLoader
from src.data.preprocessor import preprocess_dataset, DataSplit
from src.data.attack_simulator import AttackSimulator


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ROBUST_DIR = os.path.dirname(TEST_DIR)
ML_RAW_DIR = os.path.join(ROBUST_DIR, "data", "raw", "ml-1m")
AMAZON_DIR = os.path.join(ROBUST_DIR, "data", "amazon")
YELP_DIR = os.path.join(ROBUST_DIR, "data", "yelp")


# ── 1. MovieLens-1M Full-Scale Loading ─────────────────────────────────────────

def test_movielens_full_scale_loading():
    """Verify that MovieLensLoader loads complete verified dataset with Nu=6040, Ni=3952."""
    assert os.path.exists(os.path.join(ML_RAW_DIR, "ratings.dat")), f"Raw ratings.dat missing at {ML_RAW_DIR}"
    
    loader = MovieLensLoader(data_dir=ML_RAW_DIR, use_full_catalog=True)
    ratings_df, movies_df, users_df = loader.load_data()

    # Dimension and count verification
    assert len(ratings_df) == 1000209, f"Expected 1,000,209 ratings, got {len(ratings_df)}"
    assert loader.num_users == 6040, f"Expected 6,040 users, got {loader.num_users}"
    assert loader.num_items == 3952, f"Expected 3,952 item IDs, got {loader.num_items}"
    assert len(movies_df) == 3952, f"Expected 3,952 movies in catalog, got {len(movies_df)}"
    assert len(users_df) == 6040, f"Expected 6,040 users in catalog, got {len(users_df)}"

    # Active interactions checks
    assert ratings_df["user_id"].nunique() == 6040
    assert ratings_df["item_id"].nunique() == 3706  # Active rated items

    # Contiguous 0-indexed bounds checks
    assert ratings_df["user_id"].min() == 0
    assert ratings_df["user_id"].max() == 6039
    assert ratings_df["item_id"].min() == 0
    assert ratings_df["item_id"].max() <= 3951

    # Movie catalog indexing alignment
    assert movies_df["item_id"].min() == 0
    assert movies_df["item_id"].max() == 3951
    assert users_df["user_id"].min() == 0
    assert users_df["user_id"].max() == 6039


# ── 2. MovieLens-1M Zero-Leakage Temporal Splitting ──────────────────────────

def test_movielens_leave_one_out_temporal_partition():
    """Verify temporal leave-one-out partitioning preserves all 6,040 users with zero leakage."""
    loader = MovieLensLoader(data_dir=ML_RAW_DIR, use_full_catalog=True)
    ratings_df, movies_df, users_df = loader.load_data()

    split = preprocess_dataset(ratings_df, movies_df, users_df, split_strategy="leave_one_out")

    assert split.num_users == 6040
    assert split.num_items == 3952
    assert len(split.train_dict) == 6040
    assert len(split.val_dict) == 6040
    assert len(split.test_dict) == 6040

    total_train_edges = sum(len(items) for items in split.train_dict.values())
    total_val_edges = len(split.val_dict)
    total_test_edges = len(split.test_dict)

    # Exact interaction accounting
    assert total_train_edges + total_val_edges + total_test_edges == 1000209
    assert total_train_edges == 1000209 - 12080  # Exactly 988,129 train edges

    # Rigorous spot-check of temporal leave-one-out order and zero leakage
    check_users = list(range(0, 200))
    for u in check_users:
        train_records = split.train_dict[u]
        val_item, _, val_ts = split.val_dict[u]
        test_item, _, test_ts = split.test_dict[u]

        train_items = set(i for i, _, _ in train_records)
        train_timestamps = [ts for _, _, ts in train_records]

        # Strict chronological ordering: train <= val <= test
        assert max(train_timestamps) <= val_ts, f"User {u} train timestamp exceeds val timestamp"
        assert val_ts <= test_ts, f"User {u} val timestamp exceeds test timestamp"

        # Zero evaluation target leakage into training set
        assert test_item not in train_items, f"User {u} test item {test_item} leaked into train_dict"
        assert val_item not in train_items, f"User {u} val item {val_item} leaked into train_dict"
        assert val_item != test_item, f"User {u} val item and test item collide"


# ── 3. Amazon Reviews 5-Core Properties ───────────────────────────────────────

def test_amazon_5core_dataset_properties():
    """Verify Amazon Reviews data loader satisfies venue-standard 5-core scale and invariants."""
    assert os.path.exists(os.path.join(AMAZON_DIR, "Electronics_5.json")), f"Amazon data missing at {AMAZON_DIR}"

    loader = AmazonReviewLoader(data_dir=AMAZON_DIR, category="Electronics", min_user_interactions=5, min_item_interactions=5)
    ratings_df, items_df, users_df = loader.load_data()

    assert not ratings_df.empty
    # Benchmark scale assertions
    assert loader.num_users >= 1000, f"Expected >= 1000 users, got {loader.num_users}"
    assert loader.num_items >= 1000, f"Expected >= 1000 items, got {loader.num_items}"
    assert len(ratings_df) >= 20000, f"Expected >= 20000 ratings, got {len(ratings_df)}"

    # 5-core interaction guarantee: every user and item must have >= 5 interactions
    user_counts = ratings_df["user_id"].value_counts()
    item_counts = ratings_df["item_id"].value_counts()
    assert (user_counts >= 5).all(), f"Min user interactions {user_counts.min()} < 5"
    assert (item_counts >= 5).all(), f"Min item interactions {item_counts.min()} < 5"

    # Contiguous 0-indexed mappings
    assert ratings_df["user_id"].min() == 0
    assert ratings_df["user_id"].max() == loader.num_users - 1
    assert ratings_df["item_id"].min() == 0
    assert ratings_df["item_id"].max() == loader.num_items - 1

    # Check temporal split integrity
    split = preprocess_dataset(ratings_df, items_df, users_df)
    assert split.num_users == loader.num_users
    assert split.num_items == loader.num_items
    assert len(split.train_dict) == loader.num_users

    # ── Text Diversity & Semantic Authenticity Assertions ───────────────────
    assert "review_text" in ratings_df.columns, "review_text column missing from ratings_df"
    n_unique_texts = ratings_df["review_text"].nunique()
    total_ratings = len(ratings_df)

    # Assert at least 1,000 unique review text strings (permanently blocks 10-sentence pools)
    assert n_unique_texts >= 1000, f"Review text diversity violation: only {n_unique_texts} unique texts (expected >= 1000)"

    # Assert review text uniqueness ratio >= 0.50 (genuine datasets typically achieve > 0.90)
    text_diversity_ratio = n_unique_texts / total_ratings
    assert text_diversity_ratio >= 0.50, f"Repetitive text violation: diversity ratio {text_diversity_ratio:.4f} < 0.50"

    # Assert no blank or whitespace-only review texts
    assert (ratings_df["review_text"].str.strip() != "").all(), "Detected empty review texts"

    # Spot-check rating-sentiment consistency (sample 100 1-star reviews)
    one_star_reviews = ratings_df[ratings_df["rating"] == 1]["review_text"].tolist()[:100]
    positive_phrases = ["excellent build quality", "crystal clear sound", "fantastic addition", "outstanding battery life"]
    for text in one_star_reviews:
        lower = text.lower()
        for phrase in positive_phrases:
            assert phrase not in lower, f"Contradictory rating-text pair detected: 1-star review contains '{phrase}'"


# ── 4. Yelp Academic 5-Core Properties ────────────────────────────────────────

def test_yelp_5core_dataset_properties():
    """Verify Yelp Academic data loader satisfies venue-standard 5-core scale and invariants."""
    assert os.path.exists(os.path.join(YELP_DIR, "yelp_academic_dataset_review.json")), f"Yelp review data missing at {YELP_DIR}"

    loader = YelpLoader(data_dir=YELP_DIR, min_user_interactions=5, min_item_interactions=5)
    ratings_df, businesses_df, users_df = loader.load_data()

    assert not ratings_df.empty
    # Benchmark scale assertions
    assert loader.num_users >= 1000, f"Expected >= 1000 users, got {loader.num_users}"
    assert loader.num_items >= 1000, f"Expected >= 1000 items, got {loader.num_items}"
    assert len(ratings_df) >= 20000, f"Expected >= 20000 ratings, got {len(ratings_df)}"

    # 5-core interaction guarantee: every user and item must have >= 5 interactions
    user_counts = ratings_df["user_id"].value_counts()
    item_counts = ratings_df["item_id"].value_counts()
    assert (user_counts >= 5).all(), f"Min user interactions {user_counts.min()} < 5"
    assert (item_counts >= 5).all(), f"Min item interactions {item_counts.min()} < 5"

    # Contiguous 0-indexed mappings
    assert ratings_df["user_id"].min() == 0
    assert ratings_df["user_id"].max() == loader.num_users - 1
    assert ratings_df["item_id"].min() == 0
    assert ratings_df["item_id"].max() == loader.num_items - 1

    # Check business categories parsing
    assert not businesses_df.empty
    assert "genres" in businesses_df.columns or "categories" in businesses_df.columns

    # ── Text Diversity & Semantic Authenticity Assertions ───────────────────
    assert "review_text" in ratings_df.columns, "review_text column missing from ratings_df"
    n_unique_texts = ratings_df["review_text"].nunique()
    total_ratings = len(ratings_df)

    assert n_unique_texts >= 1000, f"Review text diversity violation: only {n_unique_texts} unique texts (expected >= 1000)"

    text_diversity_ratio = n_unique_texts / total_ratings
    assert text_diversity_ratio >= 0.50, f"Repetitive text violation: diversity ratio {text_diversity_ratio:.4f} < 0.50"

    assert (ratings_df["review_text"].str.strip() != "").all(), "Detected empty review texts"

    # Spot-check rating-sentiment consistency (sample 100 1-star reviews)
    one_star_reviews = ratings_df[ratings_df["rating"] == 1]["review_text"].tolist()[:100]
    positive_phrases = ["exceptional hospitality", "delicious small plates", "loved the ambiance", "beautifully plated"]
    for text in one_star_reviews:
        lower = text.lower()
        for phrase in positive_phrases:
            assert phrase not in lower, f"Contradictory rating-text pair detected: 1-star review contains '{phrase}'"

    # ── Business Metadata Diversity Assertions (Anti-Cloning) ─────────────────
    assert "title" in businesses_df.columns, "title column missing from businesses_df"
    n_unique_titles = businesses_df["title"].nunique()
    assert n_unique_titles >= 500, f"Business cloning violation: only {n_unique_titles} unique titles (expected >= 500)"

    # Ensure no cloned '#1', '#2' pattern appended to business names
    has_hash_clone = businesses_df["title"].str.contains(r"#\d+$").any()
    assert not has_hash_clone, "Detected cloned business titles with artificial '#<number>' pattern"


# ── 5. Full-Scale Zero-Leakage Bandwagon Attack Protocol ──────────────────────

def test_zero_leakage_bandwagon_attack_full_scale():
    """
    Verify Bandwagon shilling attack at rho=0.15 on full-scale MovieLens-1M:
      - Injects N_shill = ceil(0.15 * 6040) = 906 fake accounts
      - Resulting users: 6040 + 906 = 6946
      - Injected interactions placed exclusively in train_dict
      - val_dict and test_dict remain strictly untouched and identical to clean split
    """
    loader = MovieLensLoader(data_dir=ML_RAW_DIR, use_full_catalog=True)
    ratings_df, movies_df, users_df = loader.load_data()
    clean_split = preprocess_dataset(ratings_df, movies_df, users_df)

    sim = AttackSimulator(seed=42)
    rho = 0.15
    expected_shills = int(np.ceil(rho * clean_split.num_users))
    assert expected_shills == 906, f"Expected 906 shills, got {expected_shills}"

    attacked_split = sim.inject_attack(clean_split, attack_type="bandwagon", noise_rate=rho)

    # User count verification
    assert attacked_split.num_users == clean_split.num_users + expected_shills == 6946
    assert len(attacked_split.train_dict) == 6946

    # Strict Zero-Leakage: val_dict and test_dict are identical to clean_split
    assert len(attacked_split.val_dict) == len(clean_split.val_dict) == 6040
    assert len(attacked_split.test_dict) == len(clean_split.test_dict) == 6040
    assert attacked_split.val_dict == clean_split.val_dict
    assert attacked_split.test_dict == clean_split.test_dict

    # Bot accounts (6040..6945): present in train_dict, absent in val_dict and test_dict
    for fake_u in range(clean_split.num_users, attacked_split.num_users):
        assert fake_u in attacked_split.train_dict
        assert fake_u not in attacked_split.val_dict
        assert fake_u not in attacked_split.test_dict
        # Every fake edge is tagged with ground_truth_labels = 0
        fake_interactions = attacked_split.train_dict[fake_u]
        assert len(fake_interactions) > 0
        for item, rating, ts in fake_interactions:
            assert attacked_split.ground_truth_labels[(fake_u, item)] == 0
            assert (fake_u, item) in attacked_split.weight_dict

    # Authentic accounts (0..6039): train_dict unchanged, tagged with ground_truth_labels = 1
    sample_users = [0, 42, 100, 500, 1000, 5000, 6039]
    for auth_u in sample_users:
        assert attacked_split.train_dict[auth_u] == clean_split.train_dict[auth_u]
        for item, rating, ts in attacked_split.train_dict[auth_u]:
            assert attacked_split.ground_truth_labels[(auth_u, item)] == 1


# ── 6. Script Data Ingestion Scale & Downsampling Verification ────────────────

def test_script_downsampling_flags_and_catalog_fidelity():
    """Verify that load_domain_data in benchmark scripts respects full scale vs sample mode."""
    import run_multiseed_benchmark as rmb

    # Full scale MovieLens
    r_full, m_full, u_full = rmb.load_domain_data("movielens", dataset_scale="full")
    assert len(r_full) == 1000209
    assert len(u_full) == 6040
    assert len(m_full) == 3952

    # Sample scale MovieLens
    r_sample, m_sample, u_sample = rmb.load_domain_data("movielens", dataset_scale="sample")
    assert len(u_sample) <= 300
    assert len(r_sample) < 1000209

    # Full scale Amazon
    r_amz, i_amz, u_amz = rmb.load_domain_data("amazon", dataset_scale="full")
    assert len(r_amz) >= 20000
    assert len(u_amz) >= 1000

    # Full scale Yelp
    r_ylp, i_ylp, u_ylp = rmb.load_domain_data("yelp", dataset_scale="full")
    assert len(r_ylp) >= 20000
    assert len(u_ylp) >= 1000
