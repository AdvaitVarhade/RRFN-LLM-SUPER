"""
robust_super/tests/test_m1_5core_adversarial_stress.py
======================================================
Adversarial Stress Test Suite for Milestone M1:
Amazon Reviews and Yelp Academic 5-Core Integrity and Data Loader Robustness.

Author: teamwork_preview_challenger_m1_2 (Adversarial Challenger)
Roles: critic, specialist

Mission:
1. Rigorous empirical verification of degree distributions and power-law properties
   for Amazon Reviews (Electronics_5.json) and Yelp Academic (reviews & businesses).
2. Formal assertion of the 5-core invariant: every user degree >= 5 and every item degree >= 5.
3. Verification of leave-one-out temporal split invariants under 5-core interaction graphs.
4. Stress testing loader error handling against empty files, whitespace files, corrupt JSON,
   null/missing attributes, corrupt gzip archives, degenerate subgraphs, and path traversal.
"""

import os
import json
import gzip
import pytest
import numpy as np
import pandas as pd

from src.data.amazon_loader import AmazonReviewLoader
from src.data.yelp_loader import YelpLoader
from src.data.preprocessor import preprocess_dataset, safe_join_path

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ROBUST_DIR = os.path.dirname(TEST_DIR)
AMAZON_DIR = os.path.join(ROBUST_DIR, "data", "amazon")
YELP_DIR = os.path.join(ROBUST_DIR, "data", "yelp")


# ==============================================================================
# 1. RAW DATASET 5-CORE INTEGRITY & DEGREE DISTRIBUTION AUDITS
# ==============================================================================

def test_raw_amazon_5core_degree_distributions():
    """
    Directly parse raw Electronics_5.json without loader mediation.
    Adversarially audit degree distributions, power-law properties, and 5-core invariants.
    """
    amz_file = os.path.join(AMAZON_DIR, "Electronics_5.json")
    assert os.path.isfile(amz_file), f"Amazon dataset file not found: {amz_file}"
    assert os.path.getsize(amz_file) > 1_000_000, f"Amazon dataset file size unexpectedly small: {os.path.getsize(amz_file)} bytes"

    records = []
    with open(amz_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            rec = json.loads(line_str)
            assert "reviewerID" in rec, f"Line {line_num} missing reviewerID"
            assert "asin" in rec, f"Line {line_num} missing asin"
            assert "overall" in rec, f"Line {line_num} missing overall rating"
            assert "unixReviewTime" in rec, f"Line {line_num} missing unixReviewTime"
            records.append((rec["reviewerID"], rec["asin"], float(rec["overall"]), int(rec["unixReviewTime"])))

    df = pd.DataFrame(records, columns=["reviewerID", "asin", "overall", "unixReviewTime"])
    total_interactions = len(df)
    n_users = df["reviewerID"].nunique()
    n_items = df["asin"].nunique()

    # Scale invariants
    assert total_interactions >= 20_000, f"Expected >= 20,000 interactions, got {total_interactions}"
    assert n_users >= 1_000, f"Expected >= 1,000 users, got {n_users}"
    assert n_items >= 1_000, f"Expected >= 1,000 items, got {n_items}"

    # Strict uniqueness: no duplicate user-item edges
    duplicates = df.duplicated(subset=["reviewerID", "asin"]).sum()
    assert duplicates == 0, f"Found {duplicates} duplicate (reviewerID, asin) edges in raw Amazon dataset"

    # Degree calculations
    u_degrees = df["reviewerID"].value_counts()
    i_degrees = df["asin"].value_counts()

    # FORMAL ASSERTION: 5-core invariant
    min_u_deg = int(u_degrees.min())
    min_i_deg = int(i_degrees.min())
    assert min_u_deg >= 5, f"User 5-core violated! min user degree = {min_u_deg} < 5"
    assert min_i_deg >= 5, f"Item 5-core violated! min item degree = {min_i_deg} < 5"

    # Distribution quantiles
    q_u = u_degrees.quantile([0.0, 0.05, 0.25, 0.50, 0.75, 0.95, 1.0]).to_dict()
    q_i = i_degrees.quantile([0.0, 0.05, 0.25, 0.50, 0.75, 0.95, 1.0]).to_dict()

    assert q_u[0.0] >= 5.0
    assert q_i[0.0] >= 5.0
    # Heavy tail / power law verification: max degree is significantly larger than median
    assert q_i[1.0] > 3 * q_i[0.50], "Item degree distribution lacks heavy-tail property"
    assert q_u[1.0] > 2 * q_u[0.50], "User degree distribution lacks heavy-tail property"

    # Rating integrity: valid ratings between 1 and 5
    assert df["overall"].min() >= 1.0
    assert df["overall"].max() <= 5.0
    assert 3.5 <= df["overall"].mean() <= 4.8


def test_raw_yelp_5core_degree_distributions():
    """
    Directly parse raw yelp_academic_dataset_review.json and business.json without loader mediation.
    Adversarially audit degree distributions, power-law properties, and 5-core invariants.
    """
    rev_file = os.path.join(YELP_DIR, "yelp_academic_dataset_review.json")
    biz_file = os.path.join(YELP_DIR, "yelp_academic_dataset_business.json")
    assert os.path.isfile(rev_file), f"Yelp review file not found: {rev_file}"
    assert os.path.isfile(biz_file), f"Yelp business file not found: {biz_file}"

    # Parse reviews
    records = []
    with open(rev_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            rec = json.loads(line_str)
            assert "user_id" in rec, f"Line {line_num} missing user_id"
            assert "business_id" in rec, f"Line {line_num} missing business_id"
            assert "stars" in rec, f"Line {line_num} missing stars rating"
            records.append((rec["user_id"], rec["business_id"], float(rec["stars"])))

    df_rev = pd.DataFrame(records, columns=["user_id", "business_id", "stars"])
    total_interactions = len(df_rev)
    n_users = df_rev["user_id"].nunique()
    n_items = df_rev["business_id"].nunique()

    assert total_interactions >= 20_000, f"Expected >= 20,000 interactions, got {total_interactions}"
    assert n_users >= 1_000, f"Expected >= 1,000 users, got {n_users}"
    assert n_items >= 1_000, f"Expected >= 1,000 items, got {n_items}"

    # Strict uniqueness
    duplicates = df_rev.duplicated(subset=["user_id", "business_id"]).sum()
    assert duplicates == 0, f"Found {duplicates} duplicate (user_id, business_id) edges in raw Yelp dataset"

    # Degree calculations
    u_degrees = df_rev["user_id"].value_counts()
    i_degrees = df_rev["business_id"].value_counts()

    # FORMAL ASSERTION: 5-core invariant
    min_u_deg = int(u_degrees.min())
    min_i_deg = int(i_degrees.min())
    assert min_u_deg >= 5, f"User 5-core violated! min user degree = {min_u_deg} < 5"
    assert min_i_deg >= 5, f"Item 5-core violated! min item degree = {min_i_deg} < 5"

    # Distribution quantiles
    q_u = u_degrees.quantile([0.0, 0.05, 0.25, 0.50, 0.75, 0.95, 1.0]).to_dict()
    q_i = i_degrees.quantile([0.0, 0.05, 0.25, 0.50, 0.75, 0.95, 1.0]).to_dict()

    assert q_u[0.0] >= 5.0
    assert q_i[0.0] >= 5.0
    assert q_i[1.0] > 3 * q_i[0.50], "Item degree distribution lacks heavy-tail property"

    # Parse and check business catalog coverage
    biz_ids = set()
    with open(biz_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                b = json.loads(line)
                assert "business_id" in b
                biz_ids.add(b["business_id"])

    review_bids = set(df_rev["business_id"].unique())
    missing_bids = review_bids - biz_ids
    assert len(missing_bids) == 0, f"Found {len(missing_bids)} businesses in reviews missing from business catalog"


# ==============================================================================
# 2. LOADER MEDIATION & 0-INDEXED CONTIGUOUS GRAPH INVARIANTS
# ==============================================================================

def test_amazon_loader_live_dataset_invariants():
    """Verify AmazonReviewLoader against the actual dataset files."""
    loader = AmazonReviewLoader(data_dir=AMAZON_DIR, category="Electronics", min_user_interactions=5, min_item_interactions=5)
    ratings_df, items_df, users_df = loader.load_data()

    assert not ratings_df.empty
    assert len(ratings_df) >= 20_000
    assert loader.num_users >= 1_000
    assert loader.num_items >= 1_000

    # 0-indexed contiguous mapping invariant
    assert ratings_df["user_id"].min() == 0
    assert ratings_df["user_id"].max() == loader.num_users - 1
    assert ratings_df["user_id"].nunique() == loader.num_users
    assert ratings_df["item_id"].min() == 0
    assert ratings_df["item_id"].max() == loader.num_items - 1
    assert ratings_df["item_id"].nunique() == loader.num_items

    # 5-core verification on loader output
    u_counts = ratings_df["user_id"].value_counts()
    i_counts = ratings_df["item_id"].value_counts()
    assert (u_counts >= 5).all()
    assert (i_counts >= 5).all()

    # Metadata dataframe shapes and IDs
    assert len(users_df) == loader.num_users
    assert len(items_df) == loader.num_items
    assert (users_df["user_id"] == np.arange(loader.num_users)).all()
    assert (items_df["item_id"] == np.arange(loader.num_items)).all()


def test_yelp_loader_live_dataset_invariants():
    """Verify YelpLoader against the actual dataset files."""
    loader = YelpLoader(data_dir=YELP_DIR, min_user_interactions=5, min_item_interactions=5)
    ratings_df, businesses_df, users_df = loader.load_data()

    assert not ratings_df.empty
    assert len(ratings_df) >= 20_000
    assert loader.num_users >= 1_000
    assert loader.num_items >= 1_000

    # 0-indexed contiguous mapping invariant
    assert ratings_df["user_id"].min() == 0
    assert ratings_df["user_id"].max() == loader.num_users - 1
    assert ratings_df["user_id"].nunique() == loader.num_users
    assert ratings_df["item_id"].min() == 0
    assert ratings_df["item_id"].max() == loader.num_items - 1
    assert ratings_df["item_id"].nunique() == loader.num_items

    # 5-core verification on loader output
    u_counts = ratings_df["user_id"].value_counts()
    i_counts = ratings_df["item_id"].value_counts()
    assert (u_counts >= 5).all()
    assert (i_counts >= 5).all()

    # Metadata dataframe shapes and IDs
    assert len(users_df) == loader.num_users
    assert len(businesses_df) == loader.num_items
    assert (users_df["user_id"] == np.arange(loader.num_users)).all()
    assert (businesses_df["item_id"] == np.arange(loader.num_items)).all()


# ==============================================================================
# 3. TEMPORAL LEAVE-ONE-OUT PREPROCESSING UNDER 5-CORE INTERACTION GRAPHS
# ==============================================================================

def test_temporal_split_degree_monotonicity_amazon():
    """
    Because Amazon is a verified 5-core graph (deg(u) >= 5):
    Under leave-one-out partitioning:
      - test_dict takes 1 interaction per user
      - val_dict takes 1 interaction per user
      - train_dict takes >= 3 interactions per user
    Every single user MUST have len(train_dict[u]) >= 3.
    """
    loader = AmazonReviewLoader(data_dir=AMAZON_DIR)
    ratings_df, items_df, users_df = loader.load_data()
    split = preprocess_dataset(ratings_df, items_df, users_df, split_strategy="leave_one_out")

    assert split.num_users == loader.num_users
    assert split.num_items == loader.num_items
    assert len(split.train_dict) == loader.num_users
    assert len(split.val_dict) == loader.num_users
    assert len(split.test_dict) == loader.num_users

    # Check that EVERY user has at least 3 training items (5 - 2 = 3)
    train_lengths = [len(items) for items in split.train_dict.values()]
    assert min(train_lengths) >= 3, f"Found user with < 3 train interactions: min={min(train_lengths)}"

    # Spot-check zero leakage across sample of 100 users
    for u in range(min(100, split.num_users)):
        train_items = set(item for item, _, _ in split.train_dict[u])
        val_item, _, val_ts = split.val_dict[u]
        test_item, _, test_ts = split.test_dict[u]

        assert val_item not in train_items
        assert test_item not in train_items
        assert val_item != test_item


def test_temporal_split_degree_monotonicity_yelp():
    """
    Because Yelp is a verified 5-core graph (deg(u) >= 5):
    Under leave-one-out partitioning:
      - Every single user MUST have len(train_dict[u]) >= 3.
    """
    loader = YelpLoader(data_dir=YELP_DIR)
    ratings_df, businesses_df, users_df = loader.load_data()
    split = preprocess_dataset(ratings_df, businesses_df, users_df, split_strategy="leave_one_out")

    assert split.num_users == loader.num_users
    assert split.num_items == loader.num_items
    assert len(split.train_dict) == loader.num_users
    assert len(split.val_dict) == loader.num_users
    assert len(split.test_dict) == loader.num_users

    train_lengths = [len(items) for items in split.train_dict.values()]
    assert min(train_lengths) >= 3, f"Found user with < 3 train interactions: min={min(train_lengths)}"


# ==============================================================================
# 4. ADVERSARIAL STRESS TESTING: EMPTY, MALFORMED, AND DEGENERATE INPUTS
# ==============================================================================

def test_amazon_loader_empty_file_robustness(tmp_path):
    """
    Adversarial test: Zero-byte file or whitespace-only file.
    AmazonReviewLoader should gracefully fall back to synthetic data without crashing.
    """
    # 1. 0-byte file
    empty_file = tmp_path / "Electronics_5.json"
    empty_file.write_text("", encoding="utf-8")

    loader = AmazonReviewLoader(data_dir=str(tmp_path), category="Electronics")
    r_df, i_df, u_df = loader.load_data()
    assert not r_df.empty, "Expected fallback to synthetic data on 0-byte file"
    assert loader.num_users > 0
    assert loader.num_items > 0

    # 2. Whitespace-only file
    empty_file.write_text("   \n\n\t  \n", encoding="utf-8")
    loader_ws = AmazonReviewLoader(data_dir=str(tmp_path), category="Electronics")
    r_df_ws, _, _ = loader_ws.load_data()
    assert not r_df_ws.empty, "Expected fallback to synthetic data on whitespace-only file"


def test_amazon_loader_corrupt_malformed_json_lines(tmp_path):
    """
    Adversarial test: JSON file containing invalid syntax, truncated objects,
    non-dict JSON tokens, and missing critical fields.
    """
    bad_file = tmp_path / "Electronics_5.json"
    bad_file.write_text(
        "INVALID_SYNTAX_LINE_1\n"
        "{truncated_json\n"
        "[1, 2, 3]\n" # JSON array instead of object -> causes AttributeError
        "\"just a string\"\n" # JSON string instead of object
        "null\n" # JSON null
        "{\"reviewerID\": null, \"asin\": \"B01\", \"overall\": 5.0}\n" # null reviewer
        "{\"reviewerID\": \"U1\", \"asin\": null, \"overall\": 5.0}\n" # null asin
        "{\"reviewerID\": \"U1\", \"asin\": \"B01\", \"overall\": null}\n" # null overall
        "{\"reviewerID\": \"U1\", \"asin\": \"B01\", \"overall\": \"NOT_A_NUMBER\"}\n" # unparseable rating
        "{\"reviewerID\": \"U1\", \"asin\": \"B01\", \"overall\": 999.0}\n" # out-of-range rating (should clip to 5)
        "{\"reviewerID\": \"U1\", \"asin\": \"B01\", \"overall\": -5.0}\n" # out-of-range rating (should clip to 1)
        "\n",
        encoding="utf-8"
    )

    # Loader should skip corrupt lines and since no valid >=5 core exists, fall back to synthetic
    loader = AmazonReviewLoader(data_dir=str(tmp_path), category="Electronics", min_user_interactions=5, min_item_interactions=5)
    r_df, i_df, u_df = loader.load_data()
    assert not r_df.empty, "Loader crashed or failed to fall back on corrupted file"


def test_amazon_loader_corrupt_gzip_stream(tmp_path):
    """
    Adversarial test: Truncated or invalid gzip file (.json.gz).
    AmazonReviewLoader should catch BadGzipFile/EOFError/zlib.error and fall back to synthetic data.
    """
    bad_gz = tmp_path / "Electronics_5.json.gz"
    # Write garbage non-gzip bytes
    bad_gz.write_bytes(b"\x1f\x8b\x08\x00_CORRUPT_GZIP_STREAM_HEADER_PAYLOAD")

    loader = AmazonReviewLoader(data_dir=str(tmp_path), category="Electronics")
    r_df, i_df, u_df = loader.load_data()
    assert not r_df.empty, "Failed to handle corrupted gzip file gracefully"


def test_amazon_loader_degenerate_graph_empty_kcore(tmp_path):
    """
    Adversarial test: A file with valid JSON lines, but the graph is too sparse
    (all users have degree 1 or 2).
    Iterative k-core filtering with k=5 reduces dataframe to 0 rows.
    Loader MUST handle the empty reduced DataFrame and fall back to synthetic data cleanly.
    """
    f = tmp_path / "Electronics_5.json"
    lines = []
    # 10 users, each with only 1 interaction
    for idx in range(10):
        lines.append(json.dumps({
            "reviewerID": f"U_{idx}",
            "asin": f"I_{idx}",
            "overall": 5.0,
            "unixReviewTime": 1000000 + idx
        }))
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")

    loader = AmazonReviewLoader(data_dir=str(tmp_path), category="Electronics", min_user_interactions=5, min_item_interactions=5)
    r_df, i_df, u_df = loader.load_data()
    # Since all 10 users had degree 1 < 5, df became empty, falling back to synthetic
    assert not r_df.empty
    assert (r_df["user_id"].value_counts() >= 5).all()


def test_yelp_loader_empty_files_robustness(tmp_path):
    """
    Adversarial test: Empty review file or missing business file.
    YelpLoader should gracefully fall back to synthetic data without crashing.
    """
    # 0-byte review file
    rev_file = tmp_path / "yelp_academic_dataset_review.json"
    rev_file.write_text("", encoding="utf-8")

    loader = YelpLoader(data_dir=str(tmp_path))
    r_df, b_df, u_df = loader.load_data()
    assert not r_df.empty, "Expected fallback to synthetic data on 0-byte review file"
    assert loader.num_users > 0
    assert loader.num_items > 0


def test_yelp_loader_corrupt_business_and_reviews(tmp_path):
    """
    Adversarial test: Malformed review lines and malformed business metadata.
    """
    rev_file = tmp_path / "yelp_academic_dataset_review.json"
    rev_file.write_text(
        "INVALID_YELP_REVIEW\n"
        "{\"user_id\": null, \"business_id\": \"B1\", \"stars\": 4.0}\n"
        "{\"user_id\": \"U1\", \"business_id\": null, \"stars\": 4.0}\n"
        "{\"user_id\": \"U1\", \"business_id\": \"B1\", \"stars\": null}\n"
        "{\"user_id\": \"U1\", \"business_id\": \"B1\", \"stars\": \"NOT_A_STAR\"}\n"
        "{\"user_id\": \"U1\", \"business_id\": \"B1\", \"stars\": 5.0, \"date\": \"GARBAGE_DATE\"}\n",
        encoding="utf-8"
    )
    biz_file = tmp_path / "yelp_academic_dataset_business.json"
    biz_file.write_text(
        "CORRUPT_BIZ_JSON\n"
        "{\"business_id\": \"B1\", \"name\": \"Valid Name\", \"categories\": null}\n"
        "{\"business_id\": \"B2\", \"name\": \"Valid 2\", \"categories\": [1, 2, 3]}\n",
        encoding="utf-8"
    )

    loader = YelpLoader(data_dir=str(tmp_path), min_user_interactions=5, min_item_interactions=5)
    r_df, b_df, u_df = loader.load_data()
    # Degenerate interactions fall back to synthetic
    assert not r_df.empty
    assert (r_df["user_id"].value_counts() >= 5).all()


def test_yelp_loader_nondict_json_lines(tmp_path):
    """
    Adversarial test: YelpLoader handling non-dict JSON lines (arrays, strings, numbers).
    """
    rev_file = tmp_path / "yelp_academic_dataset_review.json"
    rev_file.write_text(
        "[1, 2, 3]\n"
        "\"just a string\"\n"
        "12345\n",
        encoding="utf-8"
    )
    loader = YelpLoader(data_dir=str(tmp_path))
    r_df, b_df, u_df = loader.load_data()
    assert not r_df.empty


def test_yelp_loader_missing_business_file_fallback(tmp_path):
    """
    Adversarial test: Reviews file exists with valid >=5 core interactions,
    but business.json is completely absent.
    YelpLoader should still successfully parse reviews and generate default business metadata.
    """
    rev_file = tmp_path / "yelp_academic_dataset_review.json"
    lines = []
    # Create 5 users, 5 items, all connected (5x5 = 25 interactions >= 5-core)
    for u in range(5):
        for b in range(5):
            lines.append(json.dumps({
                "user_id": f"USER_{u}",
                "business_id": f"BIZ_{b}",
                "stars": 4.0,
                "date": f"2021-0{u+1}-0{b+1} 12:00:00"
            }))
    rev_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Do NOT create business.json
    loader = YelpLoader(data_dir=str(tmp_path), min_user_interactions=5, min_item_interactions=5)
    r_df, b_df, u_df = loader.load_data()

    assert len(r_df) == 25
    assert loader.num_users == 5
    assert loader.num_items == 5
    # Default business names should be created
    assert len(b_df) == 5
    assert "genres" in b_df.columns
    assert "title" in b_df.columns
