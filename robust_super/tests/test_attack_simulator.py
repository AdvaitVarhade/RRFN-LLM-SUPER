import pytest
from src.data.loader import MovieLensLoader
from src.data.preprocessor import preprocess_dataset
from src.data.attack_simulator import AttackSimulator

def test_attack_simulator_all_types():
    loader = MovieLensLoader(data_dir="non_existent", min_user_interactions=2, min_item_interactions=2)
    ratings_df, movies_df, users_df = loader.load_data()
    clean_split = preprocess_dataset(ratings_df, movies_df, users_df)

    simulator = AttackSimulator(seed=42)

    # 1. Random flip
    flip_split = simulator.inject_attack(clean_split, "random_flip", noise_rate=0.20)
    assert len(flip_split.ground_truth_labels) > 0
    injected_count = sum(1 for label in flip_split.ground_truth_labels.values() if label == 0)
    assert injected_count > 0

    # 2. Bandwagon
    bandwagon_split = simulator.inject_attack(clean_split, "bandwagon", noise_rate=0.10)
    assert bandwagon_split.num_users > clean_split.num_users

    # 3. Nuke bomb
    nuke_split = simulator.inject_attack(clean_split, "nuke_bomb", noise_rate=0.10)
    assert nuke_split.num_users > clean_split.num_users

    # 4. AGAS
    agas_split = simulator.inject_attack(clean_split, "agas", noise_rate=0.10)
    assert agas_split.num_users > clean_split.num_users
