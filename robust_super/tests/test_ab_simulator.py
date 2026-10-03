import pytest
from src.evaluation.ab_simulator import (
    estimate_ctr,
    estimate_gmv_lift,
    compute_tail_exposure_score,
    generate_ab_report,
)


def test_estimate_ctr_linear_formula_and_clamping():
    # reference nDCG = 0.08 -> at 0.08, CTR = base_ctr (0.12)
    ctr_ref = estimate_ctr(0.08, base_ctr=0.12, sensitivity=1.8)
    assert pytest.approx(ctr_ref, abs=1e-6) == 0.12

    # nDCG = 0.16 -> rel_change = (0.16 - 0.08)/0.08 = 1.0 -> raw_ctr = 0.12 * (1 + 1.8 * 1) = 0.336
    ctr_high = estimate_ctr(0.16, base_ctr=0.12, sensitivity=1.8)
    assert pytest.approx(ctr_high, abs=1e-6) == 0.336

    # Clamping upper bound: nDCG = 1.0 -> should clamp at 1.0
    ctr_max = estimate_ctr(1.0, base_ctr=0.5, sensitivity=5.0)
    assert ctr_max == 1.0

    # Clamping lower bound: nDCG = 0.0 -> raw_ctr = 0.12 * (1 - 1.8) = -0.096 -> clamped to 0.0
    ctr_min = estimate_ctr(0.0, base_ctr=0.12, sensitivity=1.8)
    assert ctr_min == 0.0


def test_estimate_ctr_invalid_inputs():
    with pytest.raises(ValueError):
        estimate_ctr(-0.1)
    with pytest.raises(ValueError):
        estimate_ctr(1.5)
    with pytest.raises(ValueError):
        estimate_ctr(0.1, base_ctr=-0.05)
    with pytest.raises(ValueError):
        estimate_ctr(0.1, base_ctr=1.2)
    with pytest.raises(ValueError):
        estimate_ctr(0.1, sensitivity=-1.0)


def test_estimate_gmv_lift_positive_and_conventions():
    res = estimate_gmv_lift(
        ctr_robust=0.20,
        ctr_vanilla=0.10,
        avg_price=50.0,
        dau=10000,
        conversion_rate=0.04,
        rollout_fraction=0.5
    )

    # daily robust GMV = 10000 * 0.5 * 0.20 * 0.04 * 50.0 = 2000.0
    # daily vanilla GMV = 10000 * 0.5 * 0.10 * 0.04 * 50.0 = 1000.0
    assert pytest.approx(res["daily_gmv_robust"]) == 2000.0
    assert pytest.approx(res["daily_gmv_vanilla"]) == 1000.0
    assert pytest.approx(res["daily_lift"]) == 1000.0
    assert pytest.approx(res["monthly_lift"]) == 30000.0
    assert pytest.approx(res["annual_lift"]) == 365000.0


def test_estimate_gmv_lift_invalid_inputs():
    with pytest.raises(ValueError):
        estimate_gmv_lift(ctr_robust=-0.1, ctr_vanilla=0.1, avg_price=50.0, dau=1000, conversion_rate=0.03)
    with pytest.raises(ValueError):
        estimate_gmv_lift(ctr_robust=0.2, ctr_vanilla=0.1, avg_price=-10.0, dau=1000, conversion_rate=0.03)
    with pytest.raises(ValueError):
        estimate_gmv_lift(ctr_robust=0.2, ctr_vanilla=0.1, avg_price=50.0, dau=-5, conversion_rate=0.03)
    with pytest.raises(ValueError):
        estimate_gmv_lift(ctr_robust=0.2, ctr_vanilla=0.1, avg_price=50.0, dau=1000, conversion_rate=1.5)
    with pytest.raises(ValueError):
        estimate_gmv_lift(ctr_robust=0.2, ctr_vanilla=0.1, avg_price=50.0, dau=1000, conversion_rate=0.03, rollout_fraction=1.2)


def test_compute_tail_exposure_score():
    # standard calculation: (0.3 - 0.2) / 0.2 * 100 = 50.0
    assert pytest.approx(compute_tail_exposure_score(0.3, 0.2)) == 50.0

    # zero vanilla baseline with positive robust
    assert compute_tail_exposure_score(0.2, 0.0) == 100.0

    # zero for both
    assert compute_tail_exposure_score(0.0, 0.0) == 0.0

    # invalid inputs
    with pytest.raises(ValueError):
        compute_tail_exposure_score(-0.1, 0.2)
    with pytest.raises(ValueError):
        compute_tail_exposure_score(0.1, -0.2)


def test_generate_ab_report_keys_and_missing_metrics():
    metrics_rob = {"nDCG@10": 0.38, "APLT@10": 0.35, "LTC@10": 0.40}
    metrics_van = {"nDCG@10": 0.32, "APLT@10": 0.25, "LTC@10": 0.30}

    report = generate_ab_report(metrics_rob, metrics_van)

    expected_keys = [
        "ndcg_robust", "ndcg_vanilla",
        "ctr_robust", "ctr_vanilla", "ctr_lift_pct",
        "aplt_robust", "aplt_vanilla", "tail_exposure_lift_pct",
        "ltc_robust", "ltc_vanilla", "ltc_lift_pct",
        "daily_gmv_robust", "daily_gmv_vanilla",
        "daily_lift", "monthly_lift", "annual_lift"
    ]
    for key in expected_keys:
        assert key in report

    assert report["ctr_robust"] > report["ctr_vanilla"]
    assert report["daily_lift"] > 0

    # Test fallback metric keys without @10
    metrics_rob_fallback = {"nDCG": 0.38, "APLT": 0.35, "LTC": 0.40}
    metrics_van_fallback = {"nDCG": 0.32, "APLT": 0.25, "LTC": 0.30}
    report_fallback = generate_ab_report(metrics_rob_fallback, metrics_van_fallback)
    assert report_fallback["ndcg_robust"] == 0.38

    # Test missing required metric raises KeyError
    incomplete_rob = {"Recall@10": 0.50}
    with pytest.raises(KeyError):
        generate_ab_report(incomplete_rob, metrics_van)
