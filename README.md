# RRFN-LLM-SUPER: Adversarially Robust and Popularity-Debiased Recommendation Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**RRFN-LLM-SUPER** is an adversarial-resilient, popularity-debiased recommendation framework engineered to preserve recommendation accuracy, catalog diversity, and user-centric calibration under intense adversarial manipulation and coordinated shilling attacks (including rating-flip noise, bandwagon attacks, and target review-bombing campaigns).

The system seamlessly combines statistical noise transition estimation, semantic LLM taste-profile auditing, spatiotemporal anomaly detection, and decoupled dual-space Pareto recommendation merging.

---

## Key Highlights and Architectural Pillars

1. **SUPER Popularity Calibration**:
   - Pareto-based effective volume catalog partitioning into head ($\mathcal{H}$) and tail ($\mathcal{T}$) partitions.
   - Decoupled dual neural representations ($M_{\text{pop}}$ specialized on head popularity, $M_{\text{tail}}$ specialized on tail discovery).
   - User-specific popularity inclination calibration $\tilde{C}(u)$ with Bayesian shrinkage smoothing against global mean inclination.
   - Soft blueprint item merging for multi-objective, Pareto-efficient Top-$N$ recommendation rankings.

2. **RRFN Statistical Transition Modeling**:
   - 5-class discrete rating transition modeling over ordinal interaction scores.
   - High-confidence anchor point selection discovering clean interaction anchors.
   - Noise transition matrix $\hat{T}$ estimation with Frobenius-regularized slack correction $\Delta T$:
     $$T_{\text{final}} = \text{Softmax}(\hat{T} + \Delta T), \quad \mathcal{L}_{\text{risk}} = \ell_{\text{risk}} + \lambda_\Delta \|\Delta T\|_F^2$$

3. **Lightweight LLM Semantic Profile Auditor**:
   - Evaluates user historical taste coherence versus candidate interaction ratings ($R_{\text{LLM}}$).
   - Compute-conserving two-stage gating: only statistically anomalous interactions ($R_{\text{RRFN}} < 0.60$) undergo LLM semantic audit.
   - SQLite prompt caching and asynchronous evaluation with Google GenAI (`gemini-2.5-flash`) and deterministic local heuristics.

4. **Spatiotemporal Review-Bombing Detector**:
   - Quantifies behavioral burst acceleration $A(i, t)$, polarity concentration $S(i, t)$, and temporal anomaly clustering ($R_{\text{bomb}}$) to identify organized shilling attacks.

5. **Multi-View Reliability Fusion and Importance Reweighting**:
   - Interaction reliability scoring synthesizing statistical, semantic, and temporal dimensions:
     $$w(u, i) = \alpha \cdot R_{\text{RRFN}}(u, i) + \beta \cdot R_{\text{LLM}}(u, i) + \gamma \cdot R_{\text{bomb}}(i, t)$$
   - Continuous sample weights $w(u, i) \in [0.02, 1.0]$ driving risk-consistent loss minimization, Pareto catalog partitioning, and blueprint denoising.

6. **Multi-Backbone and Multi-Domain Support**:
   - Graph Collaborative Filtering: LightGCN, SimGCL, and SGL.
   - Deep Hybrid and Autoencoder Architectures: NeuMF and VaeCF.
   - Multi-Domain evaluation across MovieLens-1M, Amazon Review (Instant Video, Digital Music), and Yelp.

---

## Comprehensive Head-to-Head Benchmark Results

Evaluating the **Base Model (Vanilla SUPER under Bandwagon Shilling Attack $\rho=0.10$)** versus **RRFN-LLM-SUPER** on MovieLens-1M:

| Category | Metric | Clean Baseline | Base Model (Attacked) | RRFN-LLM-SUPER (Ours) | Relative Gain (%) | Result |
|---|---|---|---|---|---|---|
| **Ranking Accuracy** | `Recall@10` ($\uparrow$) | `0.4910` | `0.4210` | **`0.4820`** | `+14.5%` | [WIN] Substantial recovery |
| | `nDCG@10` ($\uparrow$) | `0.3840` | `0.3250` | **`0.3780`** | `+16.3%` | [WIN] Higher ranking quality |
| | `Precision@10` ($\uparrow$) | `0.3340` | `0.2860` | **`0.3280`** | `+14.7%` | [WIN] Consistent accuracy |
| | `HitRate@10` ($\uparrow$) | `0.7420` | `0.6580` | **`0.7350`** | `+11.7%` | [WIN] High hit frequency |
| **Popularity Calibration** | `RMSE-PC` ($\downarrow$) | `0.0780` | `0.1420` | **`0.0820`** | `+42.3%` | [WIN] 42% lower error |
| | `MRMC` ($\downarrow$) | `0.0910` | `0.1650` | **`0.0940`** | `+43.0%` | [WIN] 43% lower miscalibration |
| **Fairness & Discovery** | `APLT@10 (%)` ($\uparrow$) | `32.0%` | `24.5%` | **`32.4%`** | `+32.2%` | [WIN] Preserved tail exposure |
| | `LTC@10 (%)` ($\uparrow$) | `41.2%` | `28.6%` | **`40.8%`** | `+42.7%` | [WIN] Diverse long-tail reach |
| | `Entropy` ($\uparrow$) | `0.8420` | `0.6810` | **`0.8350`** | `+22.6%` | [WIN] Broad item coverage |
| | `Novelty (bits)` ($\uparrow$) | `12.85` | `10.40` | **`12.72`** | `+22.3%` | [WIN] Serendipitous discovery |
| **Overall Performance** | `GKPI` ($\uparrow$) | `0.6350` | `0.4420` | **`0.6280`** | `+42.1%` | [WIN] Composite index preserved |
| **Adversarial Robustness** | `Delta-GKPI (%)` ($\downarrow$) | `0.0%` | `+30.4%` | **`+1.1%`** | `+96.4%` | [WIN] 96% less degradation |
| | `CSS` ($\downarrow$) | `0.0000` | `0.0640` | **`0.0040`** | `+93.8%` | [WIN] High ranking stability |
| | `Denoising-F1` ($\uparrow$) | `1.0000` | `0.0000` | **`0.9120`** | `+100.0%` | [WIN] Accurate attack rejection |
| | `Denoising-ROC-AUC` ($\uparrow$) | `1.0000` | `0.5000` | **`0.9450`** | `+89.0%` | [WIN] Strong discrimination |

---

## Repository Structure

```
├── robust_super/
│   ├── app.py                         # Interactive 8-Tab Streamlit Web Platform
│   ├── run_dashboard.bat              # One-Click Windows Streamlit Launcher
│   ├── configs/
│   │   └── default_config.yaml        # Master Hyperparameter and Dataset Configuration
│   ├── src/
│   │   ├── data/                      # Loaders (MovieLens, Amazon, Yelp), Attack Simulator
│   │   ├── models/                    # NeuMF, LightGCN, SimGCL, SGL, VaeCF Neural Backbones
│   │   ├── rrfn/                      # Anchor discovery, Transition Matrix, Risk Loss
│   │   ├── llm_auditor/               # LLM Prompt Builder, SQLite Cache, Live & Mock Auditor
│   │   ├── bombing_detector/          # Temporal Density, Polarity Skew, Anomaly Scorer
│   │   ├── fusion/                    # Multi-View Reliability Fusion (Alpha, Beta, Gamma)
│   │   ├── catalog/                   # Pareto Catalog Partitioning (Head vs Tail)
│   │   ├── super_engine/              # User Inclination, Soft Blueprint, Decoupled Merger
│   │   ├── trainer/                   # Decoupled Dual Trainer, Learning Rate Scheduler
│   │   └── evaluation/                # 13 Evaluation Metrics, A/B Simulator, LaTeX Exporters
│   ├── experiments/
│   │   ├── run_baseline.py            # Clean SUPER Baseline Runner
│   │   ├── run_attack.py              # Attacked Vanilla Benchmark
│   │   ├── run_robust_super.py        # RRFN-LLM-SUPER Benchmark
│   │   ├── run_ablations.py           # 7-Variant Systematic Ablation Runner
│   │   ├── run_comprehensive_comparison.py # Full Cross-Attack Head-to-Head Runner
│   │   └── generate_plots.py          # Academic Publication Figure Generators
│   ├── screenshots/                   # Full-Resolution Visual Evidence (Tabs 1 to 8)
│   ├── results/                       # Persisted Multi-Seed, GPU, and Domain Benchmarks
│   └── tests/                         # 85-Test Comprehensive Pytest Verification Suite
├── run_multiseed_benchmark.py         # 5-Seed Statistical Significance Verification Script
├── train_15epoch_gpu.py               # Full 15-Epoch Accelerated PyTorch Training Script
├── train_multidomain.py               # Cross-Domain SimGCL / SGL Pipeline (Amazon & Yelp)
├── print_results.py                   # Consolidated Terminal Results Formatter
├── requirements.txt                   # Production Dependencies
├── AUDIT_REPORT.md                    # Multi-Role Architecture and Security Audit Report
├── .gitignore                         # Repository Exclusion Rules
└── README.md                          # Technical Framework Documentation
```

---

## Quickstart Guide

### 1. Environment Setup and Installation

Ensure Python 3.10 or higher is installed with PyTorch support.

```bash
# Clone the repository
git clone https://github.com/AdvaitVarhade/RRFN-LLM-SUPER.git
cd RRFN-LLM-SUPER

# Install required dependencies
pip install -r requirements.txt
```

### 2. Run Test Suite

Verify all unit, numerical stability, security, and integration tests across all 11 test modules:

```bash
python -m pytest robust_super/tests/ -v
```

### 3. Launch Interactive Web Dashboard

Launch the full-featured Streamlit platform:

```bash
streamlit run robust_super/app.py
```

Or on Windows:

```cmd
robust_super\run_dashboard.bat
```

Open `http://localhost:8501` in your browser.

### 4. Run Benchmarks and Multi-Seed Validation

Execute the 5-seed statistical benchmark:

```bash
python run_multiseed_benchmark.py --seeds 42 123 456 789 2026 --epochs 15
```

Execute multi-domain contrastive graph learning benchmarks on Amazon and Yelp:

```bash
python train_multidomain.py --epochs 15
```

---

## Interactive Web Dashboard Modules

The platform features an 8-tab diagnostic and experimentation suite:

1. **Multi-View Signal Fusion**: Donut visualization of active signal weights ($\alpha, \beta, \gamma$), fused reliability distribution histograms, and review-bombing sensitivity controls.
2. **Noise Transition Matrix and Denoising Diagnostics**: Dynamic heatmap of $T_{\text{final}}$, ROC-AUC and Precision-Recall denoising curves, and dual-model convergence monitoring.
3. **Academic Benchmark and Head-to-Head Comparison**: 13-metric comparative evaluation table, multi-dimensional Radar / Spider plots, attack-budget perturbation curves ($\rho \in [0.0, 0.25]$), and automated LaTeX/CSV exports.
4. **7-Variant Systematic Ablation Study**: Comparative evaluation isolating the individual contributions of RRFN, LLM semantic auditing, review-bombing detection, and Pareto debiasing.
5. **Live LLM Semantic Auditor and Inspector**: Live Google GenAI (`gemini-2.5-flash`) integration, SQLite prompt cache telemetry, interactive prompt/response inspection, and single-interaction sandbox.
6. **Live User Profile and What-If Inclination Sandbox**: Per-user rating history with threat flags, side-by-side Top-10 recommendation lists across three systems, and an interactive inclination slider.
7. **Multi-Platform A/B Testing Simulator**: Two-sample Student t-test and Mann-Whitney U test calculations evaluating Click-Through Rate (CTR) and Long-Tail Discovery lift over 1,000 simulated users.
8. **Multi-Domain Contrastive Graph Learning**: Evaluation of SimGCL and SGL architectures across cross-domain benchmarks (Amazon Instant Video, Amazon Digital Music, Yelp).

---

## Mathematical Formulation

### 1. Risk-Consistent Noise Transition Loss
Given corrupted interaction ratings $\tilde{Y} \in \{1, \dots, K\}$, the risk-consistent loss utilizes the estimated noise transition matrix $T_{\text{final}} \in \mathbb{R}^{K \times K}$:

$$\mathcal{L}_{\text{risk}}(\hat{y}, \tilde{y}) = \ell(\hat{y}, \tilde{y}) \cdot T_{\text{final}}^{-1} + \lambda_\Delta \|\Delta T\|_F^2$$

### 2. Multi-View Reliability Fusion
For every user-item pair $(u, i)$:

$$w(u, i) = \alpha \cdot R_{\text{RRFN}}(u, i) + \beta \cdot R_{\text{LLM}}(u, i) + \gamma \cdot R_{\text{bomb}}(i, t)$$

Where $\alpha + \beta + \gamma = 1.0$ and $w(u, i) \in [0.02, 1.0]$.

### 3. Popularity Inclination Calibration
User popularity inclination $C(u)$ measures consumption bias toward head items. Bayesian smoothing prevents distortion for low-activity users:

$$\tilde{C}(u) = \frac{|I_u|}{|I_u| + \tau} C(u) + \frac{\tau}{|I_u| + \tau} \bar{C}$$

Where $\bar{C}$ represents global mean inclination and $\tau$ denotes the smoothing prior.

### 4. Generalized Key Performance Index (GKPI)
A holistic performance metric balancing ranking quality, fairness, and robustness:

$$\text{GKPI} = 0.40 \cdot \text{nDCG@10} + 0.30 \cdot (1 - \text{RMSE-PC}) + 0.20 \cdot \text{APLT@10} + 0.10 \cdot \text{Novelty}_{\text{norm}}$$

---

## Citation and References

```bibtex
@article{rrfn_llm_super_2026,
  title={RRFN-LLM-SUPER: Adversarially Robust and Popularity-Debiased Recommendation Framework},
  author={Varhade, Advait and Contributors},
  journal={IEEE Access},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
