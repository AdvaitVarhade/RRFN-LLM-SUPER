"""
A/B Business Impact Simulator for SUPER Recommendation Debiasing System.
Provides functions to estimate CTR lift, GMV (Gross Merchandise Value) impact,
and long-tail exposure metrics under simulated A/B testing conditions.
"""

from typing import Dict, Any, Optional


def estimate_ctr(
    ndcg: float,
    base_ctr: float = 0.12,
    sensitivity: float = 1.8
) -> float:
    """
    Estimates Click-Through Rate (CTR) from nDCG using the authoritative linear reference model:
        CTR = base_ctr * (1 + sensitivity * ((ndcg - reference_ndcg) / reference_ndcg))
    where reference_ndcg = 0.08.

    Args:
        ndcg: The model's nDCG score. Must be between 0.0 and 1.0.
        base_ctr: Baseline CTR (default: 0.12 or 12%). Must be in [0, 1].
        sensitivity: Sensitivity multiplier for nDCG relative change (default: 1.8).

    Returns:
        float: Estimated CTR, strictly clamped to [0.0, 1.0].

    Raises:
        ValueError: If ndcg or base_ctr are outside valid probability ranges.
    """
    if not isinstance(ndcg, (int, float)) or ndcg < 0.0 or ndcg > 1.0:
        raise ValueError(f"nDCG must be a float between 0.0 and 1.0, got: {ndcg}")
    if not isinstance(base_ctr, (int, float)) or base_ctr < 0.0 or base_ctr > 1.0:
        raise ValueError(f"base_ctr must be a float between 0.0 and 1.0, got: {base_ctr}")
    if not isinstance(sensitivity, (int, float)) or sensitivity < 0.0:
        raise ValueError(f"sensitivity must be a non-negative number, got: {sensitivity}")

    reference_ndcg = 0.08
    rel_change = (ndcg - reference_ndcg) / reference_ndcg
    raw_ctr = base_ctr * (1.0 + sensitivity * rel_change)

    # Clamp to valid CTR range [0.0, 1.0]
    clamped_ctr = max(0.0, min(1.0, float(raw_ctr)))
    return clamped_ctr


def estimate_gmv_lift(
    ctr_robust: float,
    ctr_vanilla: float,
    avg_price: float,
    dau: int,
    conversion_rate: float,
    rollout_fraction: float = 0.5
) -> Dict[str, float]:
    """
    Computes daily, monthly, and annual Gross Merchandise Value (GMV) lift
    between robust and vanilla models.

    Daily GMV formula per model variant:
        daily_gmv = DAU * rollout_fraction * CTR * conversion_rate * average_item_price

    Time conventions:
        - Monthly lift: 30 days per month
        - Annual lift: 365 days per year

    Args:
        ctr_robust: CTR of the robust model in [0, 1].
        ctr_vanilla: CTR of the vanilla model in [0, 1].
        avg_price: Average item price ($). Must be >= 0.
        dau: Daily Active Users. Must be >= 0 integer/number.
        conversion_rate: Purchase conversion rate given a click in [0, 1].
        rollout_fraction: Percentage of total DAU receiving treatment in [0, 1] (default 0.5).

    Returns:
        Dict[str, float]: Dictionary containing:
            - 'daily_gmv_robust': Daily revenue for robust model
            - 'daily_gmv_vanilla': Daily revenue for vanilla baseline
            - 'daily_lift': Daily GMV difference (robust - vanilla)
            - 'monthly_lift': 30-day GMV lift
            - 'annual_lift': 365-day GMV lift

    Raises:
        ValueError: If any input parameters are out of valid physical bounds.
    """
    if not isinstance(ctr_robust, (int, float)) or not (0.0 <= ctr_robust <= 1.0):
        raise ValueError(f"ctr_robust must be in [0.0, 1.0], got {ctr_robust}")
    if not isinstance(ctr_vanilla, (int, float)) or not (0.0 <= ctr_vanilla <= 1.0):
        raise ValueError(f"ctr_vanilla must be in [0.0, 1.0], got {ctr_vanilla}")
    if not isinstance(avg_price, (int, float)) or avg_price < 0.0:
        raise ValueError(f"avg_price must be >= 0, got {avg_price}")
    if not isinstance(dau, (int, float)) or dau < 0:
        raise ValueError(f"dau must be >= 0, got {dau}")
    if not isinstance(conversion_rate, (int, float)) or not (0.0 <= conversion_rate <= 1.0):
        raise ValueError(f"conversion_rate must be in [0.0, 1.0], got {conversion_rate}")
    if not isinstance(rollout_fraction, (int, float)) or not (0.0 <= rollout_fraction <= 1.0):
        raise ValueError(f"rollout_fraction must be in [0.0, 1.0], got {rollout_fraction}")

    daily_gmv_robust = float(dau * rollout_fraction * ctr_robust * conversion_rate * avg_price)
    daily_gmv_vanilla = float(dau * rollout_fraction * ctr_vanilla * conversion_rate * avg_price)
    daily_lift = daily_gmv_robust - daily_gmv_vanilla
    monthly_lift = daily_lift * 30.0
    annual_lift = daily_lift * 365.0

    return {
        "daily_gmv_robust": daily_gmv_robust,
        "daily_gmv_vanilla": daily_gmv_vanilla,
        "daily_lift": daily_lift,
        "monthly_lift": monthly_lift,
        "annual_lift": annual_lift,
    }


def compute_tail_exposure_score(aplt_robust: float, aplt_vanilla: float) -> float:
    """
    Computes percentage increase in Average Percentage of Long-Tail items (APLT).

    Formula:
        (APLT_robust - APLT_vanilla) / APLT_vanilla * 100

    Zero baseline behavior:
        If APLT_vanilla == 0:
            If APLT_robust > 0: Returns 100.0 (indicates positive exposure jump from 0)
            Else: Returns 0.0

    Args:
        aplt_robust: APLT score of robust model.
        aplt_vanilla: APLT score of vanilla model.

    Returns:
        float: Percentage lift in tail exposure score.
    """
    if not isinstance(aplt_robust, (int, float)) or aplt_robust < 0.0:
        raise ValueError(f"aplt_robust must be a non-negative number, got {aplt_robust}")
    if not isinstance(aplt_vanilla, (int, float)) or aplt_vanilla < 0.0:
        raise ValueError(f"aplt_vanilla must be a non-negative number, got {aplt_vanilla}")

    if aplt_vanilla == 0.0:
        return 100.0 if aplt_robust > 0.0 else 0.0

    return float((aplt_robust - aplt_vanilla) / aplt_vanilla * 100.0)


def generate_ab_report(
    metrics_robust: Dict[str, float],
    metrics_vanilla: Dict[str, float],
    sim_params: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Generates a full A/B business impact simulation report by combining evaluation metrics
    and business impact calculations.

    Looks up metric keys in order of precedence:
        nDCG: 'nDCG@10', 'nDCG'
        APLT: 'APLT@10', 'APLT'
        LTC: 'LTC@10', 'LTC'

    Args:
        metrics_robust: Metric dictionary from evaluate_recommendations for robust model.
        metrics_vanilla: Metric dictionary from evaluate_recommendations for vanilla model.
        sim_params: Optional dictionary overriding default business simulation parameters:
            - base_ctr (default: 0.12)
            - sensitivity (default: 1.8)
            - avg_price (default: 45.0)
            - dau (default: 50000)
            - conversion_rate (default: 0.03)
            - rollout_fraction (default: 0.5)

    Returns:
        Dict[str, float]: Detailed A/B test simulation results including estimated CTR,
                          GMV lifts, CTR lift percentage, and tail exposure scores.

    Raises:
        KeyError: If required recommendation metrics (nDCG, APLT, LTC) cannot be found.
    """
    def extract_metric(metrics_dict: Dict[str, float], key_options: list, name: str) -> float:
        for k in key_options:
            if k in metrics_dict:
                return float(metrics_dict[k])
        raise KeyError(f"Missing required metric '{name}'. Searched keys: {key_options}")

    ndcg_rob = extract_metric(metrics_robust, ["nDCG@10", "nDCG"], "nDCG")
    ndcg_van = extract_metric(metrics_vanilla, ["nDCG@10", "nDCG"], "nDCG")

    aplt_rob = extract_metric(metrics_robust, ["APLT@10", "APLT"], "APLT")
    aplt_van = extract_metric(metrics_vanilla, ["APLT@10", "APLT"], "APLT")

    ltc_rob = extract_metric(metrics_robust, ["LTC@10", "LTC"], "LTC")
    ltc_van = extract_metric(metrics_vanilla, ["LTC@10", "LTC"], "LTC")

    params = {
        "base_ctr": 0.12,
        "sensitivity": 1.8,
        "avg_price": 45.0,
        "dau": 50000,
        "conversion_rate": 0.03,
        "rollout_fraction": 0.5,
    }
    if sim_params:
        params.update(sim_params)

    ctr_rob = estimate_ctr(ndcg_rob, base_ctr=params["base_ctr"], sensitivity=params["sensitivity"])
    ctr_van = estimate_ctr(ndcg_van, base_ctr=params["base_ctr"], sensitivity=params["sensitivity"])

    ctr_lift_pct = ((ctr_rob - ctr_van) / ctr_van * 100.0) if ctr_van > 0 else 0.0

    gmv_results = estimate_gmv_lift(
        ctr_robust=ctr_rob,
        ctr_vanilla=ctr_van,
        avg_price=params["avg_price"],
        dau=int(params["dau"]),
        conversion_rate=params["conversion_rate"],
        rollout_fraction=params["rollout_fraction"],
    )

    tail_exposure_lift = compute_tail_exposure_score(aplt_rob, aplt_van)
    ltc_lift_pct = ((ltc_rob - ltc_van) / ltc_van * 100.0) if ltc_van > 0 else 0.0

    report = {
        "ndcg_robust": ndcg_rob,
        "ndcg_vanilla": ndcg_van,
        "ctr_robust": ctr_rob,
        "ctr_vanilla": ctr_van,
        "ctr_lift_pct": ctr_lift_pct,
        "aplt_robust": aplt_rob,
        "aplt_vanilla": aplt_van,
        "tail_exposure_lift_pct": tail_exposure_lift,
        "ltc_robust": ltc_rob,
        "ltc_vanilla": ltc_van,
        "ltc_lift_pct": ltc_lift_pct,
        **gmv_results,
    }

    return report
