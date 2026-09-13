import pytest
from src.bombing_detector.temporal_density import compute_temporal_acceleration
from src.bombing_detector.polarity_skew import compute_polarity_skew
from src.bombing_detector.semantic_similarity import compute_semantic_similarity
from src.bombing_detector.bomb_scorer import compute_bomb_scores

def test_review_bombing_detector():
    # Synthetic burst: item 0 rated 10 times in 1 hour
    train_dict = {
        u: [(0, 1, 100000 + u * 10)] for u in range(10)
    }
    train_dict[10] = [(1, 4, 500000)]

    accel = compute_temporal_acceleration(train_dict, time_window_hours=24)
    polarity = compute_polarity_skew(train_dict, time_window_hours=24)
    sim = compute_semantic_similarity(train_dict)

    bomb_scores = compute_bomb_scores(accel, polarity, sim)

    # Item 0 was bombed with 1s in a tight burst -> lower reliability
    # Item 1 was a normal isolated interaction -> higher reliability
    score_item0 = bomb_scores.get((0, 0), 0.5)
    score_item1 = bomb_scores.get((10, 1), 0.5)

    assert score_item0 < score_item1
    assert 0.0 <= score_item0 <= 1.0
    assert 0.0 <= score_item1 <= 1.0

    # Test sweep_omega_sensitivity
    from src.bombing_detector.bomb_scorer import sweep_omega_sensitivity
    gt_labels = {(0, 0): 0, (10, 1): 1}
    sweep_res = sweep_omega_sensitivity(accel, polarity, sim, gt_labels, steps=5)
    assert len(sweep_res) == 5
    assert "omega_1_temporal" in sweep_res[0]
    assert "Denoising_F1" in sweep_res[0]

