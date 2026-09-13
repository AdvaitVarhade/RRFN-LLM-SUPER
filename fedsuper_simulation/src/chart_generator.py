"""
fedsuper_simulation/src/chart_generator.py
High-Fidelity Analytical Plotly Chart Generators for Privacy-Preserving Federated SUPER.

Visualizes popularity calibration, long-tail flattening, user calibration error distributions,
multi-objective Pareto frontiers (Accuracy vs. Fairness vs. Privacy), and federated training convergence.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =============================================================================
# THEME & PALETTE CONSTANTS
# =============================================================================
COLOR_BG_PAPER = "#0E1117"          # Streamlit Core Dark Background
COLOR_BG_PLOT = "#161B22"           # Elevated Dark Plot Background
COLOR_GRID = "#30363D"              # Low-Contrast Structural Grid
COLOR_TEXT_PRIMARY = "#E6EDF3"      # High-Legibility Light Text
COLOR_TEXT_MUTED = "#8B949E"        # Muted Secondary Text

COLOR_STREAM_1_ORANGE = "#FF7043"   # Baseline Uncalibrated / Loss / Local Enclave
COLOR_STREAM_2_CYAN = "#00E5FF"     # FedSUPER Calibrated / DP Gradients
COLOR_STREAM_3_MAGENTA = "#D500F9"  # LLM Semantic Blueprints
COLOR_TARGET_GOLD = "#FFD600"       # User Target Blueprint / Operating Highlight
COLOR_SUCCESS_GREEN = "#00E676"     # Optimal / Target Threshold (CE <= 0.060)
COLOR_ACCENT_PURPLE = "#7C4DFF"     # Multi-Objective Accent

FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


# =============================================================================
# UNIFIED DARK THEME HELPER
# =============================================================================
def apply_dark_theme(
    fig: go.Figure,
    title: Optional[str] = None,
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    height: int = 420,
    show_legend: bool = True,
    legend_orientation: str = "h",
    legend_y: float = 1.12,
    margin: Optional[Dict[str, int]] = None
) -> go.Figure:
    """
    Applies the unified cyber dark layout theme to a Plotly Figure.
    """
    if margin is None:
        margin = dict(l=50, r=30, t=60 if title else 30, b=50)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLOR_BG_PAPER,
        plot_bgcolor=COLOR_BG_PLOT,
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(family=FONT_FAMILY, size=15, color=COLOR_TEXT_PRIMARY),
            x=0.02,
            y=0.96,
            xanchor="left",
            yanchor="top"
        ) if title else None,
        font=dict(family=FONT_FAMILY, color=COLOR_TEXT_PRIMARY, size=12),
        margin=margin,
        height=height,
        showlegend=show_legend,
        legend=dict(
            orientation=legend_orientation,
            yanchor="bottom" if legend_orientation == "h" else "middle",
            y=legend_y if legend_orientation == "h" else 0.5,
            xanchor="left" if legend_orientation == "h" else "left",
            x=0.0 if legend_orientation == "h" else 1.02,
            bgcolor="rgba(22, 27, 34, 0.8)",
            bordercolor=COLOR_GRID,
            borderwidth=1,
            font=dict(size=11, color=COLOR_TEXT_PRIMARY)
        ),
        hoverlabel=dict(
            bgcolor=COLOR_BG_PLOT,
            bordercolor=COLOR_GRID,
            font=dict(family=FONT_FAMILY, size=12, color=COLOR_TEXT_PRIMARY)
        )
    )

    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=COLOR_GRID,
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor=COLOR_GRID,
        title=dict(text=xaxis_title, font=dict(color=COLOR_TEXT_MUTED, size=12)) if xaxis_title else None,
        tickfont=dict(color=COLOR_TEXT_MUTED, size=11),
        linecolor=COLOR_GRID
    )

    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=COLOR_GRID,
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor=COLOR_GRID,
        title=dict(text=yaxis_title, font=dict(color=COLOR_TEXT_MUTED, size=12)) if yaxis_title else None,
        tickfont=dict(color=COLOR_TEXT_MUTED, size=11),
        linecolor=COLOR_GRID
    )

    return fig


# =============================================================================
# CHART 1: HEAD VS TORSO VS TAIL EXPOSURE GROUPED / STACKED BAR CHART
# =============================================================================
def render_exposure_distribution_chart(
    metrics: Optional[Union[Dict[str, Any], np.ndarray]] = None,
    calib_metrics: Optional[Dict[str, Any]] = None,
    theme: str = "plotly_dark",
    barmode: str = "group",
    height: int = 400
) -> go.Figure:
    """
    Renders a grouped/stacked bar chart comparing recommendation exposure shares across
    popularity tiers: Head (top 20%), Torso (next 30%), and Tail (bottom 50%).

    Compares:
      1. User Target Blueprint (P_u ground truth preference)
      2. Baseline FedAvg (Uncalibrated - heavily skewed to Head)
      3. FedSUPER Calibrated (Corrected - aligned with user target blueprint)

    Parameters:
        metrics: Dictionary returned by evaluator.evaluate_all_metrics() or array of uncalibrated counts.
        calib_metrics: Optional calibrated metrics dictionary or array.
        theme: Base Plotly template name.
        barmode: 'group' for side-by-side comparison, 'stack' for cumulative composition.
        height: Figure height in pixels.

    Returns:
        go.Figure: Interactive Plotly bar chart.
    """
    fig = go.Figure()
    tiers = ["Head (Top 20%)", "Torso (Next 30%)", "Tail (Bottom 50%)"]

    p_head, p_torso, p_tail = 0.35, 0.30, 0.35
    u_head, u_torso, u_tail = 0.82, 0.14, 0.04
    c_head, c_torso, c_tail = 0.35, 0.29, 0.36

    if isinstance(metrics, dict) and ("uncalibrated" in metrics or "calibrated" in metrics or "blueprint" in metrics):
        bp = metrics.get("blueprint", {})
        uncal = metrics.get("uncalibrated", {})
        cal = metrics.get("calibrated", {})

        p_head = float(bp.get("head_ratio", p_head))
        p_torso = float(bp.get("torso_ratio", p_torso))
        p_tail = float(bp.get("tail_ratio", p_tail))

        u_head = float(uncal.get("head_exposure_ratio", u_head))
        u_torso = float(uncal.get("torso_exposure_ratio", u_torso))
        u_tail = float(uncal.get("tail_exposure_ratio", u_tail))

        c_head = float(cal.get("head_exposure_ratio", c_head))
        c_torso = float(cal.get("torso_exposure_ratio", c_torso))
        c_tail = float(cal.get("tail_exposure_ratio", c_tail))

    elif isinstance(metrics, (list, np.ndarray)) and calib_metrics is not None:
        unc_arr = np.asarray(metrics, dtype=np.float64).flatten()
        cal_arr = np.asarray(calib_metrics, dtype=np.float64).flatten() if not isinstance(calib_metrics, dict) else np.asarray(calib_metrics.get("exposure_counts", []), dtype=np.float64).flatten()

        tot_u = max(float(np.sum(unc_arr)), 1e-6)
        tot_c = max(float(np.sum(cal_arr)), 1e-6)
        m = max(len(unc_arr), len(cal_arr), 10)
        h_idx = max(1, int(0.20 * m))
        t_idx = max(h_idx + 1, int(0.50 * m))

        u_head = float(np.sum(unc_arr[:h_idx])) / tot_u if len(unc_arr) >= h_idx else 0.80
        u_torso = float(np.sum(unc_arr[h_idx:t_idx])) / tot_u if len(unc_arr) >= t_idx else 0.15
        u_tail = float(np.sum(unc_arr[t_idx:])) / tot_u if len(unc_arr) >= t_idx else 0.05

        c_head = float(np.sum(cal_arr[:h_idx])) / tot_c if len(cal_arr) >= h_idx else 0.35
        c_torso = float(np.sum(cal_arr[h_idx:t_idx])) / tot_c if len(cal_arr) >= t_idx else 0.30
        c_tail = float(np.sum(cal_arr[t_idx:])) / tot_c if len(cal_arr) >= t_idx else 0.35

    target_pcts = [p_head * 100.0, p_torso * 100.0, p_tail * 100.0]
    uncal_pcts = [u_head * 100.0, u_torso * 100.0, u_tail * 100.0]
    cal_pcts = [c_head * 100.0, c_torso * 100.0, c_tail * 100.0]

    # Trace 1: Target Blueprint (Gold)
    fig.add_trace(go.Bar(
        name="Target Blueprint (P_u)",
        x=tiers,
        y=target_pcts,
        marker=dict(
            color=COLOR_TARGET_GOLD,
            line=dict(color="rgba(255, 214, 0, 0.8)", width=1.5)
        ),
        text=[f"{v:.1f}%" for v in target_pcts],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Target Blueprint: %{y:.1f}%<extra></extra>"
    ))

    # Trace 2: Baseline FedAvg Uncalibrated (Orange)
    fig.add_trace(go.Bar(
        name="Baseline FedAvg (Uncalibrated)",
        x=tiers,
        y=uncal_pcts,
        marker=dict(
            color=COLOR_STREAM_1_ORANGE,
            line=dict(color="rgba(255, 112, 67, 0.8)", width=1.5)
        ),
        text=[f"{v:.1f}%" for v in uncal_pcts],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Baseline Exposure: %{y:.1f}%<extra></extra>"
    ))

    # Trace 3: FedSUPER Calibrated (Cyan)
    fig.add_trace(go.Bar(
        name="FedSUPER (Calibrated)",
        x=tiers,
        y=cal_pcts,
        marker=dict(
            color=COLOR_STREAM_2_CYAN,
            line=dict(color="rgba(0, 229, 255, 0.8)", width=1.5)
        ),
        text=[f"{v:.1f}%" for v in cal_pcts],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>FedSUPER Exposure: %{y:.1f}%<extra></extra>"
    ))

    fig.update_layout(
        barmode=barmode,
        bargap=0.20,
        bargroupgap=0.08
    )

    apply_dark_theme(
        fig,
        title="Popularity Exposure Distribution by Item Tier",
        xaxis_title="Item Popularity Catalog Tier",
        yaxis_title="Share of Total Recommendations (%)",
        height=height
    )
    max_val = max(max(uncal_pcts), max(cal_pcts), max(target_pcts), 10.0)
    fig.update_yaxes(range=[0, max_val * 1.25])
    return fig


# Backward compatibility alias
render_exposure_comparison_chart = render_exposure_distribution_chart


# =============================================================================
# CHART 2: LOG-SCALE RANK VS EXPOSURE DISTRIBUTION CURVE
# =============================================================================
def render_long_tail_flattening_chart(
    exposure_uncalib: Optional[Union[np.ndarray, List[int], Dict[str, Any]]] = None,
    exposure_calib: Optional[Union[np.ndarray, List[int]]] = None,
    head_cutoff: Optional[int] = None,
    torso_cutoff: Optional[int] = None,
    theme: str = "plotly_dark",
    log_y: bool = False,
    height: int = 400
) -> go.Figure:
    """
    Renders the item rank vs. recommendation exposure distribution curve.
    Highlights the 'long-tail flattening' effect of FedSUPER calibration,
    eliminating the steep power-law exposure cliff of uncalibrated collaborative filtering.

    Parameters:
        exposure_uncalib: (M,) array of item exposure counts for baseline or full metrics dict.
        exposure_calib: (M,) array of item exposure counts for FedSUPER.
        head_cutoff: Item rank dividing Head and Torso (default: 20% of catalog).
        torso_cutoff: Item rank dividing Torso and Tail (default: 50% of catalog).
        theme: Base Plotly template name.
        log_y: Whether to render Y-axis on logarithmic scale.
        height: Figure height in pixels.

    Returns:
        go.Figure: Interactive Plotly line/area chart.
    """
    if isinstance(exposure_uncalib, dict):
        uncal_dict = exposure_uncalib.get("uncalibrated", {})
        cal_dict = exposure_uncalib.get("calibrated", {})
        exp_unc = np.asarray(uncal_dict.get("exposure_counts", []), dtype=np.float64).flatten()
        exp_cal = np.asarray(cal_dict.get("exposure_counts", []), dtype=np.float64).flatten()
    else:
        exp_unc = np.asarray(exposure_uncalib if exposure_uncalib is not None else [], dtype=np.float64).flatten()
        exp_cal = np.asarray(exposure_calib if exposure_calib is not None else [], dtype=np.float64).flatten()

    # Fallback dummy curves if empty
    if len(exp_unc) == 0 and len(exp_cal) == 0:
        m = 100
        ranks = np.arange(1, m + 1)
        exp_unc = 1000.0 / (ranks ** 1.3)
        exp_cal = 200.0 / (ranks ** 0.35)
    else:
        m = max(len(exp_unc), len(exp_cal), 10)
        if len(exp_unc) < m:
            exp_unc = np.pad(exp_unc, (0, m - len(exp_unc)))
        if len(exp_cal) < m:
            exp_cal = np.pad(exp_cal, (0, m - len(exp_cal)))

        # Sort descending to plot pure rank-frequency distribution
        exp_unc = np.sort(exp_unc)[::-1]
        exp_cal = np.sort(exp_cal)[::-1]
        ranks = np.arange(1, m + 1)

    if head_cutoff is None:
        head_cutoff = max(1, int(round(0.20 * m)))
    if torso_cutoff is None:
        torso_cutoff = max(head_cutoff + 1, int(round(0.50 * m)))

    fig = go.Figure()

    # Trace 1: Uncalibrated Baseline (Steep Cliff)
    fig.add_trace(go.Scatter(
        x=ranks,
        y=exp_unc,
        mode="lines",
        name="Baseline FedAvg (Steep Cliff)",
        line=dict(color=COLOR_STREAM_1_ORANGE, width=2.5),
        fill="tozeroy",
        fillcolor="rgba(255, 112, 67, 0.12)",
        hovertemplate="Rank #%{x}<br>Baseline Exposures: %{y:.0f}<extra></extra>"
    ))

    # Trace 2: FedSUPER Calibrated (Flattened Long Tail)
    fig.add_trace(go.Scatter(
        x=ranks,
        y=exp_cal,
        mode="lines",
        name="FedSUPER (Flattened Tail)",
        line=dict(color=COLOR_STREAM_2_CYAN, width=2.5),
        fill="tozeroy",
        fillcolor="rgba(0, 229, 255, 0.15)",
        hovertemplate="Rank #%{x}<br>FedSUPER Exposures: %{y:.0f}<extra></extra>"
    ))

    # Vertical Tier Divider Shapes & Annotations
    fig.add_vline(
        x=head_cutoff,
        line=dict(color=COLOR_TARGET_GOLD, width=1.5, dash="dash"),
        annotation_text="Head / Torso Split",
        annotation_position="top left",
        annotation_font=dict(size=10, color=COLOR_TARGET_GOLD)
    )

    fig.add_vline(
        x=torso_cutoff,
        line=dict(color=COLOR_STREAM_3_MAGENTA, width=1.5, dash="dash"),
        annotation_text="Torso / Tail Split",
        annotation_position="top left",
        annotation_font=dict(size=10, color=COLOR_STREAM_3_MAGENTA)
    )

    apply_dark_theme(
        fig,
        title="Item Popularity Rank vs. Recommendation Exposure Curve",
        xaxis_title="Item Popularity Rank (Sorted Descending)",
        yaxis_title="Recommendation Exposure Count" + (" (Log Scale)" if log_y else ""),
        height=height
    )

    if log_y:
        fig.update_yaxes(type="log")

    return fig


# Backward compatibility alias
render_rank_exposure_curve = render_long_tail_flattening_chart


# =============================================================================
# CHART 3: USER CALIBRATION ERROR DISTRIBUTION (BOX / VIOLIN)
# =============================================================================
def render_calibration_error_chart(
    ce_uncalib: Optional[Union[np.ndarray, List[float], Dict[str, Any]]] = None,
    ce_calib: Optional[Union[np.ndarray, List[float]]] = None,
    plot_type: str = "box",
    theme: str = "plotly_dark",
    height: int = 400
) -> go.Figure:
    """
    Renders a Box or Violin plot comparing per-user Calibration Error distributions (Rmse-PC).

    Demonstrates that FedSUPER reduces mean calibration error and compresses
    error variance across all client cohorts.

    Parameters:
        ce_uncalib: (U,) array of user CE values or comprehensive metrics dict.
        ce_calib: (U,) array of user CE values under FedSUPER.
        plot_type: 'box' or 'violin'.
        theme: Base Plotly template name.
        height: Figure height in pixels.

    Returns:
        go.Figure: Interactive Plotly box/violin chart.
    """
    if isinstance(ce_uncalib, dict):
        uncal_dict = ce_uncalib.get("uncalibrated", {})
        cal_dict = ce_uncalib.get("calibrated", {})
        u_ce = np.asarray(uncal_dict.get("per_user_ce", []), dtype=np.float64).flatten()
        c_ce = np.asarray(cal_dict.get("per_user_ce", []), dtype=np.float64).flatten()
    else:
        u_ce = np.asarray(ce_uncalib if ce_uncalib is not None else [], dtype=np.float64).flatten()
        c_ce = np.asarray(ce_calib if ce_calib is not None else [], dtype=np.float64).flatten()

    # Fallback dummy distributions if empty
    if len(u_ce) == 0:
        rng = np.random.default_rng(42)
        u_ce = rng.normal(loc=0.38, scale=0.08, size=50).clip(0.15, 0.65)
    if len(c_ce) == 0:
        rng = np.random.default_rng(42)
        c_ce = rng.normal(loc=0.04, scale=0.015, size=50).clip(0.01, 0.08)

    fig = go.Figure()

    if plot_type == "violin":
        fig.add_trace(go.Violin(
            y=u_ce,
            name="Baseline FedAvg",
            box_visible=True,
            meanline_visible=True,
            line_color=COLOR_STREAM_1_ORANGE,
            fillcolor="rgba(255, 112, 67, 0.3)",
            points="all",
            jitter=0.25,
            pointpos=-0.3,
            marker=dict(size=4, color=COLOR_STREAM_1_ORANGE),
            hovertemplate="Baseline User CE: %{y:.4f}<extra></extra>"
        ))
        fig.add_trace(go.Violin(
            y=c_ce,
            name="FedSUPER Calibrated",
            box_visible=True,
            meanline_visible=True,
            line_color=COLOR_STREAM_2_CYAN,
            fillcolor="rgba(0, 229, 255, 0.3)",
            points="all",
            jitter=0.25,
            pointpos=-0.3,
            marker=dict(size=4, color=COLOR_STREAM_2_CYAN),
            hovertemplate="FedSUPER User CE: %{y:.4f}<extra></extra>"
        ))
    else:
        fig.add_trace(go.Box(
            y=u_ce,
            name="Baseline FedAvg",
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.8,
            marker=dict(color=COLOR_STREAM_1_ORANGE, size=5),
            line=dict(color=COLOR_STREAM_1_ORANGE, width=2),
            fillcolor="rgba(255, 112, 67, 0.2)",
            hovertemplate="Baseline User CE: %{y:.4f}<extra></extra>"
        ))
        fig.add_trace(go.Box(
            y=c_ce,
            name="FedSUPER Calibrated",
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.8,
            marker=dict(color=COLOR_STREAM_2_CYAN, size=5),
            line=dict(color=COLOR_STREAM_2_CYAN, width=2),
            fillcolor="rgba(0, 229, 255, 0.2)",
            hovertemplate="FedSUPER User CE: %{y:.4f}<extra></extra>"
        ))

    # Reference benchmark threshold line at Rmse-PC = 0.060
    fig.add_hline(
        y=0.060,
        line=dict(color=COLOR_SUCCESS_GREEN, width=1.5, dash="dot"),
        annotation_text="SUPER Benchmark Target (CE ≤ 0.060)",
        annotation_position="bottom right",
        annotation_font=dict(size=10, color=COLOR_SUCCESS_GREEN)
    )

    apply_dark_theme(
        fig,
        title="User Popularity Calibration Error Distribution (Rmse-PC)",
        xaxis_title="Recommendation Strategy Cohort",
        yaxis_title="Per-User Calibration Error CE(u)",
        height=height
    )

    return fig


# Backward compatibility alias
render_user_calibration_distribution = render_calibration_error_chart


# =============================================================================
# CHART 4: ACCURACY - FAIRNESS - PRIVACY MULTI-OBJECTIVE PARETO FRONTIER
# =============================================================================
def render_pareto_frontier_chart(
    pareto_records: Optional[List[Dict[str, Any]]] = None,
    current_alpha: float = 0.40,
    current_lambda: float = 0.70,
    current_epsilon: float = 4.0,
    is_3d: bool = False,
    theme: str = "plotly_dark",
    height: int = 450
) -> go.Figure:
    """
    Renders the Multi-Objective Pareto Frontier balancing:
      - Accuracy: Recommendation Loss / RMSE (lower = better) or Recall@K
      - Fairness: Popularity Calibration Error Rmse-PC (lower = better) & Gini
      - Privacy: Differential Privacy Budget Epsilon (lower = stronger privacy)

    Supports both 3D interactive scatter and 2D projection modes.

    Parameters:
        pareto_records: List of dictionaries with keys ('alpha', 'lambda', 'epsilon', 'rmse', 'rmse_pc', 'gini', 'lter')
        current_alpha: Active calibration strength to highlight
        current_lambda: Active LLM semantic fusion weight
        current_epsilon: Active DP epsilon
        is_3d: If True, renders 3D interactive scatter; if False, renders 2D projection
        theme: Base Plotly template name
        height: Figure height in pixels

    Returns:
        go.Figure: Interactive Plotly 3D or 2D scatter chart
    """
    if not pareto_records:
        pareto_records = []
        alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        epsilons = [1.0, 2.0, 4.0, 8.0]
        for eps in epsilons:
            dp_penalty = 0.15 / np.sqrt(eps)
            for a in alphas:
                base_rmse = 0.48 + dp_penalty + (0.04 * a)
                rmse_pc_val = max(0.02, 0.42 * (1.0 - a) ** 1.8 + 0.035)
                gini_val = 0.78 - 0.45 * a
                lter_val = 0.03 + 0.34 * a
                pareto_records.append({
                    "alpha": float(a),
                    "lambda": 0.70,
                    "epsilon": float(eps),
                    "rmse": float(base_rmse),
                    "rmse_pc": float(rmse_pc_val),
                    "gini": float(gini_val),
                    "lter": float(lter_val)
                })

    df = pd.DataFrame(pareto_records)

    # Ensure required columns exist with safe defaults
    if "rmse" not in df.columns:
        df["rmse"] = 0.50
    if "rmse_pc" not in df.columns:
        df["rmse_pc"] = df.get("gini", 0.05)
    if "epsilon" not in df.columns:
        df["epsilon"] = 4.0
    if "alpha" not in df.columns:
        df["alpha"] = 0.40
    if "lter" not in df.columns:
        df["lter"] = 0.30

    fig = go.Figure()

    if is_3d:
        # 3D Interactive Scatter: X=RMSE, Y=Rmse-PC, Z=Epsilon
        fig.add_trace(go.Scatter3d(
            x=df["rmse"],
            y=df["rmse_pc"],
            z=df["epsilon"],
            mode="markers",
            marker=dict(
                size=6,
                color=df["alpha"],
                colorscale="Viridis",
                colorbar=dict(
                    title="Calib α",
                    titleside="top",
                    tickcolor=COLOR_TEXT_MUTED,
                    tickfont=dict(color=COLOR_TEXT_MUTED, size=10),
                    len=0.75
                ),
                opacity=0.85
            ),
            text=[f"α={a:.2f}, ε={e:.1f}" for a, e in zip(df["alpha"], df["epsilon"])],
            hovertemplate=(
                "<b>Pareto Point</b><br>"
                "RMSE: %{x:.3f}<br>"
                "Rmse-PC: %{y:.3f}<br>"
                "Epsilon (ε): %{z:.1f}<br>"
                "<extra></extra>"
            ),
            name="Tradeoff Surface"
        ))

        # Highlight current operational point
        curr_mask = (np.isclose(df["alpha"], current_alpha, atol=0.10)) & (np.isclose(df["epsilon"], current_epsilon, atol=0.50))
        if curr_mask.any():
            curr_row = df[curr_mask].iloc[0]
            fig.add_trace(go.Scatter3d(
                x=[curr_row["rmse"]],
                y=[curr_row["rmse_pc"]],
                z=[curr_row["epsilon"]],
                mode="markers+text",
                marker=dict(size=10, color=COLOR_STREAM_2_CYAN, symbol="diamond", line=dict(color="#FFFFFF", width=2)),
                text=["Selected Point"],
                textposition="top center",
                name="Current Config"
            ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLOR_BG_PAPER,
            margin=dict(l=20, r=20, t=50, b=20),
            height=height,
            title=dict(
                text="<b>Accuracy - Fairness - Privacy 3D Pareto Space</b>",
                font=dict(family=FONT_FAMILY, size=15, color=COLOR_TEXT_PRIMARY)
            ),
            scene=dict(
                xaxis=dict(title="Accuracy (RMSE)", backgroundcolor=COLOR_BG_PLOT, gridcolor=COLOR_GRID),
                yaxis=dict(title="Fairness (Rmse-PC)", backgroundcolor=COLOR_BG_PLOT, gridcolor=COLOR_GRID),
                zaxis=dict(title="Privacy (ε Budget)", backgroundcolor=COLOR_BG_PLOT, gridcolor=COLOR_GRID),
                bgcolor=COLOR_BG_PLOT
            )
        )
        return fig

    # 2D Projection Mode: Accuracy (RMSE) vs Fairness (Rmse-PC)
    fig.add_trace(go.Scatter(
        x=df["rmse"],
        y=df["rmse_pc"],
        mode="markers",
        marker=dict(
            size=10,
            color=df["alpha"],
            colorscale=[[0.0, COLOR_STREAM_1_ORANGE], [0.5, COLOR_ACCENT_PURPLE], [1.0, COLOR_STREAM_2_CYAN]],
            colorbar=dict(
                title="Calib α",
                titleside="top",
                tickfont=dict(color=COLOR_TEXT_MUTED, size=10),
                len=0.85
            ),
            showscale=True,
            line=dict(color="rgba(255,255,255,0.4)", width=1)
        ),
        text=[f"α={r['alpha']:.2f}, ε={r['epsilon']:.1f}, LTER={r['lter']:.2f}" for _, r in df.iterrows()],
        hovertemplate=(
            "<b>Configuration</b><br>"
            "Accuracy (RMSE): %{x:.3f}<br>"
            "Fairness (Rmse-PC): %{y:.3f}<br>"
            "%{text}<extra></extra>"
        ),
        name="Evaluated Configs"
    ))

    # Compute and plot Pareto Non-Dominated Frontier Line
    sorted_df = df.sort_values(by="rmse")
    pareto_pts = []
    min_rmse_pc = float("inf")
    for _, row in sorted_df.iterrows():
        if row["rmse_pc"] < min_rmse_pc:
            pareto_pts.append((row["rmse"], row["rmse_pc"]))
            min_rmse_pc = row["rmse_pc"]

    if len(pareto_pts) > 1:
        px, py = zip(*pareto_pts)
        fig.add_trace(go.Scatter(
            x=list(px),
            y=list(py),
            mode="lines",
            line=dict(color=COLOR_SUCCESS_GREEN, width=2, dash="dash"),
            name="Pareto Frontier (Non-Dominated)",
            hoverinfo="skip"
        ))

    # Highlight active operational point
    curr_point = df.iloc[(df["alpha"] - current_alpha).abs().argsort()[:1]]
    if not curr_point.empty:
        r = curr_point.iloc[0]
        fig.add_trace(go.Scatter(
            x=[r["rmse"]],
            y=[r["rmse_pc"]],
            mode="markers+text",
            marker=dict(size=14, color=COLOR_TARGET_GOLD, symbol="star", line=dict(color="#000000", width=1.5)),
            text=["Active (α=" + f"{current_alpha:.2f})"],
            textposition="top right",
            name="Active Point"
        ))

    apply_dark_theme(
        fig,
        title="Accuracy vs. Fairness Multi-Objective Pareto Frontier",
        xaxis_title="Recommendation Error (RMSE - Lower is Better)",
        yaxis_title="Popularity Calibration Error (Rmse-PC - Lower is Better)",
        height=height
    )

    return fig


# Backward compatibility alias
render_pareto_front_chart = render_pareto_frontier_chart


# =============================================================================
# CHART 5: FEDERATED TRAINING CONVERGENCE HISTORY
# =============================================================================
def render_training_convergence_chart(
    history_records: Optional[List[Dict[str, Any]]] = None,
    theme: str = "plotly_dark",
    height: int = 420
) -> go.Figure:
    """
    Renders dual-axis training convergence curves across federated communication rounds:
      - Primary Y-Axis (Left): Global BPR Loss / Training Loss (Monotonic decrease)
      - Secondary Y-Axis (Right): Popularity Calibration Error (Rmse-PC) or Catalog Update %

    Parameters:
        history_records: List of round dictionaries from FederatedSimulation.history
        theme: Base Plotly template name
        height: Figure height in pixels

    Returns:
        go.Figure: Interactive Plotly dual-axis line chart
    """
    if not history_records:
        # Fallback dummy history for round 0 / preview
        history_records = [
            {"round": r, "loss": float(0.68 * np.exp(-0.15 * r) + 0.12), "items_touched_pct": 35.0}
            for r in range(1, 11)
        ]

    rounds = [int(r.get("round", i + 1)) for i, r in enumerate(history_records)]
    losses = [float(r.get("loss", 0.0)) for r in history_records]

    fig = make_subplots(
        specs=[[{"secondary_y": True}]],
        figure=go.Figure()
    )

    # Primary Trace: Global BPR Loss (Orange)
    fig.add_trace(
        go.Scatter(
            x=rounds,
            y=losses,
            mode="lines+markers",
            name="Global BPR Loss",
            line=dict(color=COLOR_STREAM_1_ORANGE, width=3),
            marker=dict(size=6, color=COLOR_STREAM_1_ORANGE),
            hovertemplate="Round %{x}<br>BPR Loss: %{y:.4f}<extra></extra>"
        ),
        secondary_y=False
    )

    # Secondary Trace: Items Touched % or Active Telemetry
    if history_records and "items_touched_pct" in history_records[0]:
        touched = [float(r.get("items_touched_pct", 0.0)) for r in history_records]
        fig.add_trace(
            go.Scatter(
                x=rounds,
                y=touched,
                mode="lines+markers",
                name="Catalog Touched (%)",
                line=dict(color=COLOR_STREAM_2_CYAN, width=2, dash="dot"),
                marker=dict(size=5, color=COLOR_STREAM_2_CYAN),
                hovertemplate="Round %{x}<br>Catalog Touched: %{y:.1f}%<extra></extra>"
            ),
            secondary_y=True
        )

    # Optional Trace: Rmse-PC Convergence if recorded
    if history_records and "rmse_pc" in history_records[0]:
        rmse_pc_hist = [float(r.get("rmse_pc", 0.0)) for r in history_records]
        fig.add_trace(
            go.Scatter(
                x=rounds,
                y=rmse_pc_hist,
                mode="lines+markers",
                name="Calibration Error (Rmse-PC)",
                line=dict(color=COLOR_SUCCESS_GREEN, width=2.5),
                marker=dict(size=6, color=COLOR_SUCCESS_GREEN),
                hovertemplate="Round %{x}<br>Rmse-PC: %{y:.4f}<extra></extra>"
            ),
            secondary_y=True
        )

    apply_dark_theme(
        fig,
        title="Federated Training Convergence & Telemetry History",
        xaxis_title="Federated Communication Round",
        yaxis_title="Global Client BPR Loss",
        height=height
    )

    fig.update_yaxes(
        title=dict(text="Catalog Update / Calibration Metric", font=dict(color=COLOR_TEXT_MUTED, size=12)),
        secondary_y=True,
        showgrid=False
    )

    return fig


# Backward compatibility alias
render_convergence_chart = render_training_convergence_chart
