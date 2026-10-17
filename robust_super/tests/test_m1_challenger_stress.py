"""
robust_super/tests/test_m1_challenger_stress.py
================================================
Adversarial Stress Test Suite for Milestone M1 (Empirical Challenger):
- Full-scale MovieLens-1M dataset partitioning & interaction conservation.
- Boundary attack rates (rho=0.0, rho=0.15, rho=0.30) across all attack topologies.
- Formal zero-leakage assertions into validation and test sets.
- Strict catalog boundary and item degree accounting.
- Deep copy and memory isolation guarantees.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.data.loader import MovieLensLoader
from src.data.preprocessor import preprocess_dataset, DataSplit
from src.data.attack_simulator import AttackSimulator

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ROBUST_DIR = os.path.dirname(TEST_DIR)
ML_RAW_DIR = os.path.join(ROBUST_DIR, "data", "raw", "ml-1m")


@pytest.fixture(scope="module")
def full_movielens_clean_split():
    """Module-level fixture to load and split full MovieLens-1M once for fast stress testing."""
    loader = MovieLensLoader(data_dir=ML_RAW_DIR, use_full_catalog=True)
    ratings_df, movies_df, users_df = loader.load_data()
    split = preprocess_dataset(ratings_df, movies_df, users_df, split_strategy="leave_one_out")
    return ratings_df, movies_df, users_df, split


# ==============================================================================
# 1. INTERACTION CONSERVATION & CHRONOLOGICAL INVARIANTS AT FULL SCALE
# ==============================================================================

def test_full_scale_interaction_conservation_and_invariants(full_movielens_clean_split):
    """
    Formal assertion of interaction conservation on full MovieLens-1M:
    - Raw interactions = 1,000,209.
    - Exactly 6,040 active users, 3,952 item IDs (3,706 active rated items).
    - Under leave-one-out:
        * len(val_dict) == 6,040
        * len(test_dict) == 6,040
        * sum(len(items) for items in train_dict.values()) == 988,129
        * Total interactions = 988,129 + 6,040 + 6,040 == 1,000,209.
    """
    ratings_df, movies_df, users_df, split = full_movielens_clean_split

    # Interaction counts
    total_ratings = len(ratings_df)
    assert total_ratings == 1000209, f"Expected 1,000,209 ratings, got {total_ratings}"

    assert split.num_users == 6040
    assert split.num_items == 3952
    assert len(split.train_dict) == 6040
    assert len(split.val_dict) == 6040
    assert len(split.test_dict) == 6040

    train_interactions = sum(len(items) for items in split.train_dict.values())
    val_interactions = len(split.val_dict)
    test_interactions = len(split.test_dict)

    # Formal conservation equality
    assert train_interactions + val_interactions + test_interactions == total_ratings, (
        f"Interaction conservation violated! {train_interactions} + {val_interactions} + {test_interactions} "
        f"= {train_interactions + val_interactions + test_interactions} != {total_ratings}"
    )
    assert train_interactions == 988129
    assert val_interactions == 6040
    assert test_interactions == 6040

    # User dictionary partition sizes: len(train_dict) + len(val_dict) + len(test_dict)
    assert len(split.train_dict) + len(split.val_dict) + len(split.test_dict) == 18120

    # Check temporal ordering & strict zero-overlap on authentic accounts
    for u in range(0, 500):
        t_items = [i for i, _, _ in split.train_dict[u]]
        t_timestamps = [ts for _, _, ts in split.train_dict[u]]
        v_item, _, v_ts = split.val_dict[u]
        te_item, _, te_ts = split.test_dict[u]

        assert max(t_timestamps) <= v_ts, f"User {u} train timestamp exceeds val timestamp"
        assert v_ts <= te_ts, f"User {u} val timestamp exceeds test timestamp"

        # Item uniqueness across splits per user
        assert v_item not in t_items, f"User {u} val item leaked into train"
        assert te_item not in t_items, f"User {u} test item leaked into train"
        assert v_item != te_item, f"User {u} val and test items collide"


# ==============================================================================
# 2. BOUNDARY ATTACK RATE RHO = 0.0 (LOWER BOUNDARY)
# ==============================================================================

@pytest.mark.parametrize("attack_type", ["bandwagon", "nuke_bomb", "agas", "random_flip", "none"])
def test_boundary_attack_rate_rho_zero(full_movielens_clean_split, attack_type):
    """
    Stress-test lower boundary attack rate rho = 0.0 across all attack topologies.
    - Zero fake users injected.
    - Exactly 6,040 users retained.
    - Train interactions strictly equal 988,129.
    - val_dict and test_dict 100% identical to clean split.
    - Zero poisoned edges (all ground_truth_labels == 1).
    """
    _, _, _, clean_split = full_movielens_clean_split
    sim = AttackSimulator(seed=101)

    attacked = sim.inject_attack(clean_split, attack_type=attack_type, noise_rate=0.0)

    assert attacked.num_users == 6040
    assert len(attacked.train_dict) == 6040
    assert len(attacked.val_dict) == 6040
    assert len(attacked.test_dict) == 6040

    train_edges = sum(len(items) for items in attacked.train_dict.values())
    assert train_edges == 988129

    # Identity check with clean split
    assert attacked.val_dict == clean_split.val_dict
    assert attacked.test_dict == clean_split.test_dict

    # Zero fake users or poisoned labels
    for u in range(6040):
        for item, _, _ in attacked.train_dict[u]:
            assert attacked.ground_truth_labels[(u, item)] == 1


# ==============================================================================
# 3. BOUNDARY ATTACK RATE RHO = 0.15 (STANDARD BANDWAGON INJECTION)
# ==============================================================================

def test_boundary_attack_rate_rho_015_bandwagon_full_scale(full_movielens_clean_split):
    """
    Stress-test standard protocol rho = 0.15 on full-scale MovieLens-1M:
    - N_shill = ceil(0.15 * 6040) = 906 fake accounts.
    - Total users = 6040 + 906 = 6946.
    - Injected interactions placed exclusively in train_dict.
    - Zero fake users or attack edges in val_dict or test_dict.
    - Ground truth labels: authentic = 1, shill = 0.
    """
    _, _, _, clean_split = full_movielens_clean_split
    sim = AttackSimulator(seed=42)

    attacked = sim.inject_attack(clean_split, attack_type="bandwagon", noise_rate=0.15)

    expected_shills = int(np.ceil(0.15 * 6040))  # 906
    assert expected_shills == 906
    assert attacked.num_users == 6946
    assert len(attacked.train_dict) == 6946

    # 1. Validation & Test Zero-Leakage Checks
    assert len(attacked.val_dict) == 6040
    assert len(attacked.test_dict) == 6040
    assert set(attacked.val_dict.keys()) == set(range(6040))
    assert set(attacked.test_dict.keys()) == set(range(6040))
    assert attacked.val_dict == clean_split.val_dict
    assert attacked.test_dict == clean_split.test_dict

    # 2. Assert zero fake user ID (>= 6040) ever touches val or test
    for fake_u in range(6040, 6946):
        assert fake_u not in attacked.val_dict, f"Fake user {fake_u} leaked into val_dict!"
        assert fake_u not in attacked.test_dict, f"Fake user {fake_u} leaked into test_dict!"

    # 3. Authentic users unchanged in train_dict
    for u in range(0, 100):
        assert attacked.train_dict[u] == clean_split.train_dict[u]
        for item, _, _ in attacked.train_dict[u]:
            assert attacked.ground_truth_labels[(u, item)] == 1

    # 4. Injected fake user edges validation
    total_fake_edges = 0
    for fake_u in range(6040, 6946):
        fake_interactions = attacked.train_dict[fake_u]
        assert len(fake_interactions) > 0, f"Fake user {fake_u} has empty interaction list"
        total_fake_edges += len(fake_interactions)
        for item, rating, ts in fake_interactions:
            # Valid item boundaries
            assert 0 <= item < 3952, f"Item {item} out of catalog bounds"
            assert 1 <= rating <= 5, f"Rating {rating} invalid"
            assert attacked.ground_truth_labels[(fake_u, item)] == 0
            assert (fake_u, item) in attacked.weight_dict
            assert attacked.weight_dict[(fake_u, item)] == 1.0

    assert total_fake_edges > 0
    # Recomputed item counts must match train_dict
    for fake_u in range(6040, 6946):
        for item, _, _ in attacked.train_dict[fake_u]:
            assert attacked.item_counts[item] >= 1


# ==============================================================================
# 4. BOUNDARY ATTACK RATE RHO = 0.30 (HIGH INTENSITY BOUNDARY)
# ==============================================================================

def test_boundary_attack_rate_rho_030_bandwagon_full_scale(full_movielens_clean_split):
    """
    Stress-test high boundary attack rate rho = 0.30 on full-scale MovieLens-1M:
    - N_shill = ceil(0.30 * 6040) = 1,812 fake accounts.
    - Total users = 6040 + 1812 = 7852.
    - All 1,812 fake users placed strictly in train_dict.
    - val_dict and test_dict remain exactly 6,040 clean entries with zero contamination.
    """
    _, _, _, clean_split = full_movielens_clean_split
    sim = AttackSimulator(seed=999)

    attacked = sim.inject_attack(clean_split, attack_type="bandwagon", noise_rate=0.30)

    expected_shills = int(np.ceil(0.30 * 6040))  # 1812
    assert expected_shills == 1812
    assert attacked.num_users == 7852
    assert len(attacked.train_dict) == 7852

    # Zero leakage checks
    assert len(attacked.val_dict) == 6040
    assert len(attacked.test_dict) == 6040
    assert attacked.val_dict == clean_split.val_dict
    assert attacked.test_dict == clean_split.test_dict

    # Check all 1812 fake users
    for fake_u in range(6040, 7852):
        assert fake_u in attacked.train_dict
        assert fake_u not in attacked.val_dict
        assert fake_u not in attacked.test_dict
        for item, rating, ts in attacked.train_dict[fake_u]:
            assert 0 <= item < 3952
            assert attacked.ground_truth_labels[(fake_u, item)] == 0
            assert (fake_u, item) in attacked.weight_dict


# ==============================================================================
# 5. ALL ATTACK TYPES AT RHO = 0.15 AND RHO = 0.30
# ==============================================================================

@pytest.mark.parametrize("attack_type", ["nuke_bomb", "agas"])
@pytest.mark.parametrize("rho,expected_shills", [(0.15, 906), (0.30, 1812)])
def test_boundary_rates_nuke_and_agas_zero_leakage(full_movielens_clean_split, attack_type, rho, expected_shills):
    """
    Stress-test Nuke Bombing and AGAS across boundary rates rho=0.15 and rho=0.30.
    Assert zero validation/test contamination and strict catalog fidelity.
    """
    _, _, _, clean_split = full_movielens_clean_split
    sim = AttackSimulator(seed=777)

    attacked = sim.inject_attack(clean_split, attack_type=attack_type, noise_rate=rho)

    assert attacked.num_users == 6040 + expected_shills
    assert len(attacked.train_dict) == 6040 + expected_shills
    assert len(attacked.val_dict) == 6040
    assert len(attacked.test_dict) == 6040
    assert attacked.val_dict == clean_split.val_dict
    assert attacked.test_dict == clean_split.test_dict

    # Spot-check fake users
    for fake_u in range(6040, 6040 + expected_shills, 50):
        assert fake_u not in attacked.val_dict
        assert fake_u not in attacked.test_dict
        assert fake_u in attacked.train_dict
        for item, _, _ in attacked.train_dict[fake_u]:
            assert 0 <= item < 3952
            assert attacked.ground_truth_labels[(fake_u, item)] == 0
            assert (fake_u, item) in attacked.weight_dict


# ==============================================================================
# 6. MUTATION ISOLATION & CLONE RIGOR
# ==============================================================================

def test_split_mutation_isolation(full_movielens_clean_split):
    """
    Verify that injecting an attack does not mutate the original clean DataSplit in place.
    """
    _, _, _, clean_split = full_movielens_clean_split
    orig_train_len = len(clean_split.train_dict)
    orig_user_count = clean_split.num_users

    sim = AttackSimulator(seed=42)
    attacked = sim.inject_attack(clean_split, "bandwagon", 0.15)

    assert clean_split.num_users == orig_user_count == 6040
    assert len(clean_split.train_dict) == orig_train_len == 6040
    assert attacked.num_users == 6946

    # Mutate attacked train_dict to prove memory independence
    attacked.train_dict[0].append((9999, 5, 999999))
    assert (9999, 5, 999999) not in clean_split.train_dict[0]
