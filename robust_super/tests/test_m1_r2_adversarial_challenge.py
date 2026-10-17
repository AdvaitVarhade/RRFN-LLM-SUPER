r"""
robust_super/tests/test_m1_r2_adversarial_challenge.py
======================================================
Adversarial Stress Test Suite for Milestone M1 (Round 2 Remediation Review):
Authentic Amazon Reviews and Yelp Academic 5-Core Benchmark Verification.

Author: teamwork_preview_challenger_m1_r2_1 (Empirical Challenger)
Roles: critic, specialist

Mission Verification Objectives:
1. Degree distributions for Amazon and Yelp: assert min user degree >= 5 and
   min item/business degree >= 5 across 100% of nodes.
2. Formally assert review text diversity: assert len(set(texts)) / len(texts) > 0.90
   on both datasets (rejecting any pool repetition).
3. Formally assert Yelp business title uniqueness: assert len(set(names)) == len(names)
   and assert that no names contain r"#\d+$" clone patterns.
4. Comprehensive edge uniqueness, referential integrity, and chronological ordering.
"""

import os
import json
import re
from collections import Counter
import numpy as np
import pytest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ROBUST_DIR = os.path.dirname(TEST_DIR)
AMAZON_DIR = os.path.join(ROBUST_DIR, "data", "amazon")
YELP_DIR = os.path.join(ROBUST_DIR, "data", "yelp")


# ==============================================================================
# 1. AMAZON 5-CORE EMPIRICAL DEGREE DISTRIBUTIONS & HEAVY TAILS
# ==============================================================================

def test_empirical_amazon_5core_degree_distributions():
    """
    Empirically verify 100% adherence to min user degree >= 5 and min item degree >= 5
    on authentic Amazon Electronics_5.json without synthetic downsampling.
    """
    amz_file = os.path.join(AMAZON_DIR, "Electronics_5.json")
    assert os.path.isfile(amz_file), f"Amazon dataset file not found: {amz_file}"
    assert os.path.getsize(amz_file) > 1_000_000, f"File unexpectedly small: {os.path.getsize(amz_file)} bytes"

    users = []
    items = []
    edges = set()
    ratings = []

    with open(amz_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            rec = json.loads(line_str)
            u = rec["reviewerID"]
            i = rec["asin"]
            r = float(rec["overall"])

            assert (u, i) not in edges, f"Duplicate edge ({u}, {i}) detected at line {line_num}"
            edges.add((u, i))
            users.append(u)
            items.append(i)
            ratings.append(r)

    total_ratings = len(users)
    assert total_ratings >= 20_000, f"Expected >= 20,000 ratings, got {total_ratings}"

    user_counts = Counter(users)
    item_counts = Counter(items)

    assert len(user_counts) >= 1_000, f"Expected >= 1,000 users, got {len(user_counts)}"
    assert len(item_counts) >= 1_000, f"Expected >= 1,000 items, got {len(item_counts)}"

    # 100% node degree verification
    u_violations = [u for u, deg in user_counts.items() if deg < 5]
    i_violations = [i for i, deg in item_counts.items() if deg < 5]

    assert len(u_violations) == 0, f"User 5-core violated for {len(u_violations)} users: {u_violations[:5]}"
    assert len(i_violations) == 0, f"Item 5-core violated for {len(i_violations)} items: {i_violations[:5]}"

    min_u_deg = min(user_counts.values())
    min_i_deg = min(item_counts.values())
    assert min_u_deg >= 5, f"Min user degree {min_u_deg} < 5"
    assert min_i_deg >= 5, f"Min item degree {min_i_deg} < 5"

    # Quantile & power-law check
    i_degrees = np.array(list(item_counts.values()))
    u_degrees = np.array(list(user_counts.values()))
    assert np.max(i_degrees) > 3 * np.median(i_degrees), "Amazon item degree distribution lacks heavy-tail property"
    assert np.max(u_degrees) > 2 * np.median(u_degrees), "Amazon user degree distribution lacks heavy-tail property"


# ==============================================================================
# 2. AMAZON REVIEW TEXT DIVERSITY (ASSERT > 0.90)
# ==============================================================================

def test_empirical_amazon_review_text_diversity():
    """
    Formally assert review text diversity: assert len(set(texts)) / len(texts) > 0.90
    on Amazon Reviews, rejecting any dummy or template sentence cycling.
    """
    amz_file = os.path.join(AMAZON_DIR, "Electronics_5.json")
    texts = []

    with open(amz_file, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            rec = json.loads(line_str)
            text = str(rec.get("reviewText", "")).strip()
            assert len(text) > 0, "Encountered blank review text in Amazon dataset"
            texts.append(text)

    total_texts = len(texts)
    unique_texts = len(set(texts))
    diversity_ratio = unique_texts / total_texts

    assert total_texts >= 20_000, f"Insufficient review texts: {total_texts}"
    assert diversity_ratio > 0.90, (
        f"Amazon text diversity ratio {diversity_ratio:.6f} <= 0.90 "
        f"({unique_texts} unique out of {total_texts})"
    )


# ==============================================================================
# 3. YELP 5-CORE EMPIRICAL DEGREE DISTRIBUTIONS & HEAVY TAILS
# ==============================================================================

def test_empirical_yelp_5core_degree_distributions():
    """
    Empirically verify 100% adherence to min user degree >= 5 and min item degree >= 5
    on authentic Yelp Academic review dataset without synthetic downsampling.
    """
    rev_file = os.path.join(YELP_DIR, "yelp_academic_dataset_review.json")
    assert os.path.isfile(rev_file), f"Yelp review file not found: {rev_file}"

    users = []
    businesses = []
    edges = set()

    with open(rev_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            rec = json.loads(line_str)
            u = rec["user_id"]
            b = rec["business_id"]

            assert (u, b) not in edges, f"Duplicate edge ({u}, {b}) detected at line {line_num}"
            edges.add((u, b))
            users.append(u)
            businesses.append(b)

    total_ratings = len(users)
    assert total_ratings >= 20_000, f"Expected >= 20,000 ratings, got {total_ratings}"

    user_counts = Counter(users)
    biz_counts = Counter(businesses)

    assert len(user_counts) >= 1_000, f"Expected >= 1,000 users, got {len(user_counts)}"
    assert len(biz_counts) >= 1_000, f"Expected >= 1,000 businesses, got {len(biz_counts)}"

    u_violations = [u for u, deg in user_counts.items() if deg < 5]
    b_violations = [b for b, deg in biz_counts.items() if deg < 5]

    assert len(u_violations) == 0, f"Yelp user 5-core violated for {len(u_violations)} users: {u_violations[:5]}"
    assert len(b_violations) == 0, f"Yelp business 5-core violated for {len(b_violations)} businesses: {b_violations[:5]}"

    min_u_deg = min(user_counts.values())
    min_b_deg = min(biz_counts.values())
    assert min_u_deg >= 5, f"Min Yelp user degree {min_u_deg} < 5"
    assert min_b_deg >= 5, f"Min Yelp business degree {min_b_deg} < 5"

    b_degrees = np.array(list(biz_counts.values()))
    assert np.max(b_degrees) > 3 * np.median(b_degrees), "Yelp business degree distribution lacks heavy-tail property"


# ==============================================================================
# 4. YELP REVIEW TEXT DIVERSITY (ASSERT > 0.90)
# ==============================================================================

def test_empirical_yelp_review_text_diversity():
    """
    Formally assert review text diversity: assert len(set(texts)) / len(texts) > 0.90
    on Yelp Academic dataset, rejecting any pool repetition.
    """
    rev_file = os.path.join(YELP_DIR, "yelp_academic_dataset_review.json")
    texts = []

    with open(rev_file, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            rec = json.loads(line)
            text = str(rec.get("text", "")).strip()
            assert len(text) > 0, "Encountered blank review text in Yelp dataset"
            texts.append(text)

    total_texts = len(texts)
    unique_texts = len(set(texts))
    diversity_ratio = unique_texts / total_texts

    assert total_texts >= 20_000, f"Insufficient review texts: {total_texts}"
    assert diversity_ratio > 0.90, (
        f"Yelp text diversity ratio {diversity_ratio:.6f} <= 0.90 "
        f"({unique_texts} unique out of {total_texts})"
    )


# ==============================================================================
# 5. YELP BUSINESS TITLE UNIQUENESS & ANTI-CLONING
# ==============================================================================

def test_empirical_yelp_business_title_uniqueness_and_anti_cloning():
    r"""
    Formally assert Yelp business title uniqueness: assert len(set(names)) == len(names)
    and assert that no names contain r"#\d+$" clone patterns.
    """
    biz_file = os.path.join(YELP_DIR, "yelp_academic_dataset_business.json")
    assert os.path.isfile(biz_file), f"Yelp business file not found: {biz_file}"

    names = []
    clone_regex = re.compile(r"#\d+$")
    clones = []

    with open(biz_file, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            rec = json.loads(line_str)
            name = str(rec["name"]).strip()
            names.append(name)
            if clone_regex.search(name):
                clones.append(name)

    total_businesses = len(names)
    unique_names = len(set(names))

    # Anti-cloning check
    assert len(clones) == 0, f"Detected {len(clones)} cloned business titles with '#<num>' pattern: {clones[:5]}"

    # Formal uniqueness assertion mandated by mission
    counts = Counter(names)
    duplicates = {n: c for n, c in counts.items() if c > 1}
    assert unique_names == total_businesses, (
        f"Yelp business title uniqueness violated: {unique_names} unique titles != {total_businesses} total businesses. "
        f"Duplicate titles detected: {duplicates}"
    )


# ==============================================================================
# 6. YELP REFERENTIAL INTEGRITY & CHRONOLOGICAL INVARIANTS
# ==============================================================================

def test_empirical_yelp_referential_integrity():
    """Verify 100% referential integrity between business catalog and review interactions."""
    biz_file = os.path.join(YELP_DIR, "yelp_academic_dataset_business.json")
    rev_file = os.path.join(YELP_DIR, "yelp_academic_dataset_review.json")

    catalog_bids = set()
    with open(biz_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                catalog_bids.add(json.loads(line)["business_id"])

    review_bids = set()
    with open(rev_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                review_bids.add(json.loads(line)["business_id"])

    missing_from_catalog = review_bids - catalog_bids
    missing_from_reviews = catalog_bids - review_bids

    assert len(missing_from_catalog) == 0, f"Reviews reference businesses not in catalog: {missing_from_catalog}"
    assert len(missing_from_reviews) == 0, f"Catalog has orphaned businesses without reviews: {missing_from_reviews}"


def test_empirical_yelp_chronological_ordering():
    """Verify that interaction timestamps per user are strictly chronologically non-decreasing."""
    rev_file = os.path.join(YELP_DIR, "yelp_academic_dataset_review.json")
    user_timestamps = {}

    with open(rev_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                user_timestamps.setdefault(rec["user_id"], []).append(rec["date"])

    for u, dates in user_timestamps.items():
        assert dates == sorted(dates), f"User {u} reviews not in chronological order: {dates[:5]}"
