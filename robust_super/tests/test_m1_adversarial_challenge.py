"""
Adversarial Stress Test Suite for Milestone M1:
Data Loader Resilience, Preprocessor Edge Cases, and AGAS Simulator Completeness.
Author: teamwork_preview_challenger_m1_2 (Empirical Challenger)
"""
import os
import pytest
import pandas as pd
import numpy as np
from src.data.loader import MovieLensLoader
from src.data.amazon_loader import AmazonReviewLoader
from src.data.yelp_loader import YelpLoader
from src.data.preprocessor import preprocess_dataset, DataSplit
from src.data.attack_simulator import AttackSimulator


# ==============================================================================
# 1. MOVIELENS LOADER ADVERSARIAL STRESS TESTS
# ==============================================================================

def test_movielens_corrupt_invalid_rating_string(tmp_path):
    """
    Stress test MovieLensLoader on ratings.dat containing 'invalid_float' rating.
    A resilient data loader must skip non-numeric rating rows rather than
    throwing unhandled ValueError in astype(int).
    """
    r_file = tmp_path / "ratings.dat"
    # Row 2 contains 4 columns separated by '::' but rating is 'invalid_float'
    r_file.write_text(
        "1::10::5::100000\n"
        "1::20::invalid_float::100001\n"
        "1::30::4::100002\n"
        "1::40::3::100003\n"
        "1::50::5::100004\n",
        encoding="latin-1"
    )
    m_file = tmp_path / "movies.dat"
    m_file.write_text(
        "10::Movie 10::Action\n"
        "20::Movie 20::Drama\n"
        "30::Movie 30::Comedy\n"
        "40::Movie 40::Action\n"
        "50::Movie 50::Sci-Fi\n",
        encoding="latin-1"
    )

    loader = MovieLensLoader(data_dir=str(tmp_path), min_user_interactions=1, min_item_interactions=1)
    
    # Adversarial check: Does loader.load_data() throw ValueError?
    # Expected resilient behavior: skip row 2, load the 4 valid rows without unhandled exception.
    try:
        r_df, m_df, u_df = loader.load_data()
        assert len(r_df) == 4, f"Expected 4 valid rows after skipping corrupt line, got {len(r_df)}"
        assert "invalid_float" not in r_df["rating"].values
    except ValueError as e:
        pytest.fail(f"MovieLensLoader threw unhandled ValueError on 'invalid_float' rating row: {e}")


def test_movielens_float_format_ratings(tmp_path):
    """
    Stress test MovieLensLoader when ratings are formatted as float strings (e.g. '4.0', '4.5').
    Direct astype(int) on string column containing '4.0' fails in Python with ValueError.
    """
    r_file = tmp_path / "ratings.dat"
    r_file.write_text(
        "1::10::5.0::100000\n"
        "1::20::4.5::100001\n"
        "1::30::3.0::100002\n"
        "1::40::2.0::100003\n"
        "1::50::5.0::100004\n",
        encoding="latin-1"
    )
    m_file = tmp_path / "movies.dat"
    m_file.write_text(
        "10::Movie 10::Action\n"
        "20::Movie 20::Drama\n"
        "30::Movie 30::Comedy\n"
        "40::Movie 40::Action\n"
        "50::Movie 50::Sci-Fi\n",
        encoding="latin-1"
    )

    loader = MovieLensLoader(data_dir=str(tmp_path), min_user_interactions=1, min_item_interactions=1)
    try:
        r_df, m_df, u_df = loader.load_data()
        assert len(r_df) == 5
    except ValueError as e:
        pytest.fail(f"MovieLensLoader threw unhandled ValueError on float-formatted rating strings: {e}")


def test_movielens_empty_lines_and_malformed_delimiters(tmp_path):
    """
    Verify MovieLensLoader ignores blank lines, comma-separated lines, and lines with incorrect column counts.
    """
    r_file = tmp_path / "ratings.dat"
    r_file.write_text(
        "\n\n"
        "1::10::5::100000\n"
        "1,10,5,100000\n"  # Wrong delimiter
        "1::20\n"           # Truncated
        "1::20::3::100001::extra_field\n"  # Too many fields
        "\n"
        "1::30::4::100002\n",
        encoding="latin-1"
    )
    m_file = tmp_path / "movies.dat"
    m_file.write_text("10::Movie 10::Action\n30::Movie 30::Comedy\n", encoding="latin-1")

    loader = MovieLensLoader(data_dir=str(tmp_path), min_user_interactions=1, min_item_interactions=1)
    r_df, m_df, u_df = loader.load_data()
    assert len(r_df) == 2
    assert loader.num_users == 1
    assert loader.num_items == 2


# ==============================================================================
# 2. AMAZON REVIEW LOADER ADVERSARIAL STRESS TESTS
# ==============================================================================

def test_amazon_loader_null_category_init(tmp_path):
    """
    Adversarial test: Pass category=None or category='' to AmazonReviewLoader.
    re.sub(r'[^a-zA-Z0-9_\\-]', '', category) must not crash with TypeError when category is None.
    """
    try:
        loader = AmazonReviewLoader(data_dir=str(tmp_path), category=None)
        assert loader.category == "Electronics"
    except TypeError as e:
        pytest.fail(f"AmazonReviewLoader crashed with TypeError when category=None: {e}")


def test_amazon_loader_corrupt_json_and_invalid_fields(tmp_path):
    """
    Stress test AmazonReviewLoader on corrupt JSON lines, truncated objects,
    null reviewer IDs, and invalid float rating strings.
    """
    loader = AmazonReviewLoader(data_dir=str(tmp_path), category="Electronics", min_user_interactions=1, min_item_interactions=1)
    f_path = tmp_path / "Electronics_5.json"
    f_path.write_text(
        "\n"
        '{"reviewerID": "U1", "asin": "I1", "overall": 5.0, "unixReviewTime": 1000}\n'
        '{"reviewerID": "U2", "asin": "I1", "overall":\n'  # Truncated JSON
        'NOT_EVEN_JSON_AT_ALL\n'
        '{"reviewerID": null, "asin": "I1", "overall": 4.0}\n'  # Null user
        '{"reviewerID": "U3", "asin": null, "overall": 4.0}\n'  # Null item
        '{"reviewerID": "U4", "asin": "I2", "overall": "invalid_float", "unixReviewTime": 1001}\n' # String float
        '{"reviewerID": "U5", "asin": "I2", "overall": 3.0, "unixReviewTime": "invalid_ts"}\n'     # Non-numeric timestamp fallback
        '{"reviewerID": "U6", "asin": "I3", "overall": 4.0, "unixReviewTime": 1003}\n'
        "\n",
        encoding="utf-8"
    )
    r_df, items_df, users_df = loader.load_data()
    # Valid lines: U1-I1 (5★), U5-I2 (3★, fallback ts), U6-I3 (4★)
    assert len(r_df) == 3
    assert set(r_df["rating"].unique()).issubset({3, 4, 5})
    assert loader.num_users == 3
    assert loader.num_items == 3


# ==============================================================================
# 3. YELP LOADER ADVERSARIAL STRESS TESTS
# ==============================================================================

def test_yelp_loader_corrupt_files_and_null_categories(tmp_path):
    """
    Stress test YelpLoader on:
    - reviews with malformed delimiters, truncated JSON, invalid rating strings, null dates
    - businesses with null categories, integer categories, malformed JSON, and empty lines.
    """
    r_file = tmp_path / "yelp_academic_dataset_review.json"
    r_file.write_text(
        "\n"
        '{"user_id": "U1", "business_id": "B1", "stars": 5.0, "date": "2020-01-01"}\n'
        '{"user_id": "U2", "business_id": "B1", "stars": "invalid_float", "date": "2020-01-02"}\n'
        'TRUNCATED_JSON_REVIEW_{"user_id": "U3"\n'
        '{"user_id": null, "business_id": "B2", "stars": 4.0}\n'
        '{"user_id": "U3", "business_id": "B2", "stars": 4.0, "date": null}\n'  # Null date -> fallback ts
        '{"user_id": "U4", "business_id": "B3", "stars": 2.0, "date": "invalid-date"}\n'  # Invalid date string
        "\n",
        encoding="utf-8"
    )
    b_file = tmp_path / "yelp_academic_dataset_business.json"
    b_file.write_text(
        '{"business_id": "B1", "name": "B1 Venue", "categories": null}\n'  # Null categories
        'CORRUPT_BUSINESS_JSON\n'
        '{"business_id": "B2", "name": "B2 Venue", "categories": 12345}\n'  # Numeric categories
        '{"business_id": "B3", "name": "B3 Venue", "categories": "Food, Coffee & Tea"}\n',
        encoding="utf-8"
    )

    loader = YelpLoader(data_dir=str(tmp_path), min_user_interactions=1, min_item_interactions=1)
    r_df, b_df, u_df = loader.load_data()

    # Valid reviews: U1-B1 (5★), U3-B2 (4★), U4-B3 (2★)
    assert len(r_df) == 3
    assert loader.num_users == 3
    assert loader.num_items == 3

    # Business categories resilience checks
    b1_row = b_df[b_df["title"] == "B1 Venue"].iloc[0]
    assert b1_row["genres"] == ["Local Business"]  # Null categories fallback

    b3_row = b_df[b_df["title"] == "B3 Venue"].iloc[0]
    assert b3_row["genres"] == ["Food", "Coffee & Tea"]


# ==============================================================================
# 4. PREPROCESS_DATASET EDGE CASES & SINGLE-USER GRAPHS
# ==============================================================================

def test_preprocess_dataset_empty_dataframe():
    """Verify preprocess_dataset on empty DataFrames (0 users, 0 ratings) raises no zero-size errors."""
    empty_r = pd.DataFrame(columns=["user_id", "item_id", "rating", "timestamp"])
    empty_m = pd.DataFrame(columns=["item_id", "title", "genres"])
    empty_u = pd.DataFrame(columns=["user_id"])

    split = preprocess_dataset(empty_r, empty_m, empty_u)
    assert split.num_users == 0
    assert split.num_items == 0
    assert split.train_dict == {}
    assert split.val_dict == {}
    assert split.test_dict == {}
    assert split.weight_dict == {}
    assert split.item_counts == {}


def test_preprocess_dataset_single_user_graph_fewer_than_3_interactions():
    """
    Under leave_one_out, users with < 3 interactions cannot populate train, val, and test.
    Verify preprocess_dataset completes without ValueError: zero-size array crashes.
    """
    # Single user with only 1 interaction
    r_df = pd.DataFrame({
        "user_id": [0],
        "item_id": [10],
        "rating": [5],
        "timestamp": [1000]
    })
    m_df = pd.DataFrame({"item_id": [10], "title": ["M10"], "genres": [["Action"]]})
    u_df = pd.DataFrame({"user_id": [0]})

    split = preprocess_dataset(r_df, m_df, u_df)
    assert split.num_users == 1
    assert split.num_items == 11  # max(item_id) + 1
    assert split.train_dict == {}
    assert split.val_dict == {}
    assert split.test_dict == {}
    assert split.weight_dict == {}


def test_preprocess_dataset_single_user_graph_sufficient_interactions():
    """
    Verify single-user graph with >= 3 interactions correctly creates train, val, and test.
    """
    # Single user with 4 interactions
    r_df = pd.DataFrame({
        "user_id": [0, 0, 0, 0],
        "item_id": [1, 2, 3, 4],
        "rating": [4, 5, 3, 5],
        "timestamp": [100, 200, 300, 400]
    })
    m_df = pd.DataFrame({"item_id": [1, 2, 3, 4], "title": [f"M{i}" for i in [1, 2, 3, 4]]})
    u_df = pd.DataFrame({"user_id": [0]})

    split = preprocess_dataset(r_df, m_df, u_df)
    assert split.num_users == 1
    assert split.num_items == 5

    # Leave-one-out: test is last (ts=400, item=4), val is second-to-last (ts=300, item=3)
    assert 0 in split.test_dict
    assert split.test_dict[0] == (4, 5, 400)
    assert 0 in split.val_dict
    assert split.val_dict[0] == (3, 3, 300)

    # Train should have remaining 2 interactions: item 1 and item 2
    assert 0 in split.train_dict
    train_items = [item for item, _, _ in split.train_dict[0]]
    assert train_items == [1, 2]

    # Every train interaction must be in weight_dict and ground_truth_labels
    assert split.weight_dict[(0, 1)] == 1.0
    assert split.weight_dict[(0, 2)] == 1.0
    assert split.ground_truth_labels[(0, 1)] == 1
    assert split.ground_truth_labels[(0, 2)] == 1


# ==============================================================================
# 5. AGAS ATTACK SIMULATOR COMPLETENESS & CONTRACT VERIFICATION
# ==============================================================================

def test_agas_attack_weight_dict_completeness_and_ratings():
    """
    Adversarial verification of AGAS (Agentic Group Shilling Attack):
    1. Every (u, i) in attacked_split.train_dict MUST exist in attacked_split.weight_dict.
    2. Genuine ratings must remain unmodified.
    3. Fake ratings must receive weight 1.0 and ground_truth_label 0.
    """
    loader = MovieLensLoader(use_synthetic=True)
    r_df, m_df, u_df = loader.load_data()
    clean_split = preprocess_dataset(r_df, m_df, u_df)

    sim = AttackSimulator(seed=123)
    attacked_split = sim.inject_attack(clean_split, "agas", noise_rate=0.15)

    assert attacked_split.num_users > clean_split.num_users, "AGAS must inject fake users"
    
    # 1. Verify weight_dict completeness across ALL train interactions
    missing_weights = []
    for u, interactions in attacked_split.train_dict.items():
        for item, rating, ts in interactions:
            if (u, item) not in attacked_split.weight_dict:
                missing_weights.append((u, item))
            else:
                assert isinstance(attacked_split.weight_dict[(u, item)], float)
                assert attacked_split.weight_dict[(u, item)] == 1.0

    assert not missing_weights, f"Found {len(missing_weights)} interactions missing from weight_dict!"

    # 2. Verify genuine interactions match original ratings
    for u, clean_interactions in clean_split.train_dict.items():
        assert u in attacked_split.train_dict
        attacked_interactions = attacked_split.train_dict[u]
        # In AGAS, genuine users' profiles are unmodified (attack only injects fake bots)
        assert clean_interactions == attacked_interactions
        for item, _, _ in clean_interactions:
            assert attacked_split.ground_truth_labels[(u, item)] == 1

    # 3. Verify injected fake users have weight 1.0 and label 0
    fake_user_ids = [u for u in attacked_split.train_dict if u >= clean_split.num_users]
    assert len(fake_user_ids) > 0, "No fake users injected by AGAS"
    for fake_u in fake_user_ids:
        interactions = attacked_split.train_dict[fake_u]
        assert len(interactions) > 0, f"Fake user {fake_u} has 0 interactions"
        for item, rating, ts in interactions:
            assert (fake_u, item) in attacked_split.weight_dict
            assert attacked_split.weight_dict[(fake_u, item)] == 1.0
            assert attacked_split.ground_truth_labels[(fake_u, item)] == 0


def test_agas_attack_on_empty_and_single_user_splits():
    """
    Verify AGAS handles edge cases: empty splits and single-user splits without crashing.
    """
    sim = AttackSimulator(seed=42)

    # Empty split
    empty_split = preprocess_dataset(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    attacked_empty = sim.inject_attack(empty_split, "agas", noise_rate=0.10)
    assert attacked_empty.num_users == 0
    assert attacked_empty.train_dict == {}

    # Single-user split with sufficient interactions
    r_df = pd.DataFrame({
        "user_id": [0, 0, 0, 0],
        "item_id": [1, 2, 3, 4],
        "rating": [4, 5, 3, 5],
        "timestamp": [100, 200, 300, 400]
    })
    m_df = pd.DataFrame({"item_id": [1, 2, 3, 4], "title": [f"M{i}" for i in [1, 2, 3, 4]]})
    u_df = pd.DataFrame({"user_id": [0]})
    single_split = preprocess_dataset(r_df, m_df, u_df)

    attacked_single = sim.inject_attack(single_split, "agas", noise_rate=0.20)
    assert attacked_single.num_users > single_split.num_users
    for u, interactions in attacked_single.train_dict.items():
        for item, rating, ts in interactions:
            assert (u, item) in attacked_single.weight_dict
            assert attacked_single.weight_dict[(u, item)] == 1.0
