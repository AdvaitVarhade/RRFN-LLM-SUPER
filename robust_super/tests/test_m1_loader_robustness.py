import os
import pytest
import pandas as pd
import numpy as np
from src.data.loader import MovieLensLoader
from src.data.amazon_loader import AmazonReviewLoader
from src.data.yelp_loader import YelpLoader
from src.data.preprocessor import preprocess_dataset, safe_join_path
from src.data.attack_simulator import AttackSimulator

def test_movielens_corrupt_lines_and_nan_resilience(tmp_path):
    """Verify MovieLensLoader skips corrupted CSV lines and calculates valid global mean rating."""
    r_file = tmp_path / "ratings.dat"
    r_file.write_text("1::10::5::100000\nMALFORMED_ROW_NO_SEPARATORS\n2::20::3::100001\n3::30::4::100002\n", encoding="latin-1")
    m_file = tmp_path / "movies.dat"
    m_file.write_text("10::Movie A::Action\n20::Movie B::Drama\n30::Movie C::Comedy\n", encoding="latin-1")

    loader = MovieLensLoader(data_dir=str(tmp_path), min_user_interactions=1, min_item_interactions=1)
    r_df, m_df, u_df = loader.load_data()

    assert len(r_df) == 3
    assert loader.num_users == 3
    assert loader.num_items == 3
    assert pytest.approx(loader.global_mean_rating, 0.01) == 4.0

def test_movielens_empty_file_reduction(tmp_path):
    """Verify MovieLensLoader returns empty DataFrames gracefully when input has 0 rows."""
    r_file = tmp_path / "ratings.dat"
    r_file.write_text("", encoding="latin-1")
    loader = MovieLensLoader(data_dir=str(tmp_path), min_user_interactions=5, min_item_interactions=5)
    r_df, m_df, u_df = loader.load_data()

    assert r_df.empty
    assert loader.num_users == 0
    assert loader.num_items == 0
    assert loader.global_mean_rating == 3.5

def test_amazon_corrupt_json_and_path_traversal(tmp_path):
    """Verify AmazonReviewLoader sanitizes category path traversal and skips bad JSON lines."""
    # Path traversal sanitization check
    loader = AmazonReviewLoader(data_dir=str(tmp_path), category="../../etc/evil_cat")
    assert ".." not in loader.category
    assert "/" not in loader.category
    assert loader.category == "etcevil_cat"

    # Malformed JSON and invalid float rating handling
    clean_loader = AmazonReviewLoader(data_dir=str(tmp_path), category="Electronics", min_user_interactions=1, min_item_interactions=1)
    f_path = tmp_path / "Electronics_5.json"
    f_path.write_text(
        '{"reviewerID": "U1", "asin": "I1", "overall": 5.0, "unixReviewTime": 1000}\n'
        'CORRUPT_JSON_LINE\n'
        '{"reviewerID": "U2", "asin": "I1", "overall": "INVALID_FLOAT", "unixReviewTime": 1001}\n'
        '{"reviewerID": "U2", "asin": "I2", "overall": 4.0, "unixReviewTime": 1002}\n',
        encoding="utf-8"
    )
    r_df, items_df, users_df = clean_loader.load_data()
    assert len(r_df) == 2
    assert set(r_df["rating"].unique()).issubset({4, 5})

def test_yelp_null_categories_and_corrupt_reviews(tmp_path):
    """Verify YelpLoader handles null categories in business file and corrupt review lines."""
    b_file = tmp_path / "yelp_academic_dataset_business.json"
    # Venue with categories: null
    b_file.write_text(
        '{"business_id": "B1", "name": "Null Cat Venue", "categories": null}\n'
        'MALFORMED_JSON_BUSINESS\n'
        '{"business_id": "B2", "name": "Normal Venue", "categories": "Restaurants, Bars"}\n',
        encoding="utf-8"
    )
    r_file = tmp_path / "yelp_academic_dataset_review.json"
    r_file.write_text(
        '{"user_id": "U1", "business_id": "B1", "stars": 5.0, "date": "2020-01-01"}\n'
        'BAD_JSON_REVIEW\n'
        '{"user_id": "U2", "business_id": "B2", "stars": 3.0, "date": "2020-01-02"}\n',
        encoding="utf-8"
    )

    loader = YelpLoader(data_dir=str(tmp_path), min_user_interactions=1, min_item_interactions=1)
    r_df, b_df, u_df = loader.load_data()

    assert not b_df.empty
    # B1 categories was null -> fallback to ["Local Business"] without AttributeError
    assert b_df.iloc[0]["genres"] == ["Local Business"]
    assert b_df.iloc[1]["genres"] == ["Restaurants", "Bars"]
    assert len(r_df) == 2

def test_preprocessor_empty_dataframe_and_split_strategy():
    """Verify preprocessor handles empty ratings without NaN crash and enforces split_strategy."""
    empty_df = pd.DataFrame()
    split = preprocess_dataset(empty_df, empty_df, empty_df)
    assert split.num_users == 0
    assert split.num_items == 0
    assert len(split.train_dict) == 0
    assert len(split.weight_dict) == 0

    with pytest.raises(ValueError, match="Unsupported split_strategy"):
        preprocess_dataset(empty_df, empty_df, empty_df, split_strategy="random_kfold")

def test_safe_join_path_guard(tmp_path):
    """Verify safe_join_path asserts subpath containment and raises ValueError on escape."""
    base = str(tmp_path)
    valid_sub = safe_join_path(base, "data", "file.json")
    assert valid_sub.startswith(base)

    # Directory traversal attempts
    with pytest.raises(ValueError, match="Path traversal detected"):
        safe_join_path(base, "../outside.txt")

    with pytest.raises(ValueError, match="Path traversal detected"):
        safe_join_path(base, "sub", "..", "..", "outside.txt")

    with pytest.raises(ValueError, match="UNC path or absolute traversal prohibited"):
        safe_join_path(base, r"\\evil-server\share\payload.dat")

    with pytest.raises(ValueError, match="UNC path or absolute traversal prohibited"):
        safe_join_path(base, "//evil-server/share/payload.dat")

def test_agas_weight_dict_completeness_and_empty_guard():
    """Verify AGAS injects weight_dict for all fake interactions and empty split does not crash."""
    loader = MovieLensLoader(data_dir="", use_synthetic=True)
    r_df, m_df, u_df = loader.load_data()
    split = preprocess_dataset(r_df, m_df, u_df)

    sim = AttackSimulator(seed=42)
    agas_split = sim.inject_attack(split, "agas", noise_rate=0.10)

    # Contract verification: every (u, i) in train_dict MUST exist in weight_dict
    for u, interactions in agas_split.train_dict.items():
        for item, rating, ts in interactions:
            assert (u, item) in agas_split.weight_dict, f"Missing weight for ({u}, {item})"
            assert agas_split.weight_dict[(u, item)] == 1.0

    # Empty split reduction guard
    empty_split = preprocess_dataset(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    res_agas = sim.inject_attack(empty_split, "agas", noise_rate=0.10)
    assert res_agas.num_users == 0
    res_band = sim.inject_attack(empty_split, "bandwagon", noise_rate=0.10)
    assert res_band.num_users == 0
    res_nuke = sim.inject_attack(empty_split, "nuke_bomb", noise_rate=0.10)
    assert res_nuke.num_users == 0
