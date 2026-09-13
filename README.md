# RRFN-LLM-SUPER: Adversarially Robust & Popularity-Debiased Recommendation Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**RRFN-LLM-SUPER** is an adversarial-resilient, popularity-debiased recommendation platform designed to maintain fairness, catalog coverage, and ranking accuracy even under severe adversarial rating corruption and coordinated attacks (including rating-flip noise, bandwagon shilling, and nuke review bombing).

---

## 🌟 Key Highlights & Architectural Pillars

1. **SUPER Popularity Calibration (IEEE Access 2026)**:
   - Pareto-based effective volume catalog partitioning ($\mathcal{H}$ Head vs $\mathcal{T}$ Tail).
   - Decoupled dual neural modeling ($M_{pop}$ on Head, $M_{tail}$ on Tail).
   - User-specific popularity inclination calibration $\tilde{C}(u)$ with Bayesian shrinkage smoothing.
   - Soft blueprint item merging for multi-objective Pareto-efficient recommendations.

2. **RRFN Statistical Transition Modeling**:
   - 5-class rating probability modeling.
   - Anchor point selection on high-confidence interactions.
   - Noise transition matrix $\hat{T}$ estimation with Frobenius-regularized slack correction $\Delta T$:
     $$T_{\text{final}} = \text{Softmax}(\hat{T} + \Delta T), \quad \mathcal{L}_{\text{risk}} = \ell_{\text{risk}} + \lambda_\Delta \|\Delta T\|_F^2$$

3. **Lightweight LLM Semantic Profile Auditor**:
   - Evaluates user historical taste coherence vs candidate ratings ($R_{\text{LLM}}$).
   - Pre-filters candidates to send only statistically suspicious interactions ($R_{\text{RRFN}} < 0.60$) to the LLM to conserve compute.
   - SQLite prompt caching and asynchronous evaluation with Google GenAI (`gemini-2.5-flash`) or local heuristic engines.

4. **Spatiotemporal Review-Bombing Detector**:
   - Behavioral burst acceleration $A(i, t)$, polarity concentration $S(i, t)$, and semantic review clustering ($R_{\text{bomb}}$).

5. **Multi-View Reliability Fusion & Importance Reweighting**:
   - Interaction reliability scoring:
     $$w(u, i) = \alpha \cdot R_{\text{RRFN}}(u, i) + \beta \cdot R_{\text{LLM}}(u, i) + \gamma \cdot R_{\text{bomb}}(i, t)$$
   - Continuous weights $w(u, i) \in [0.02, 1.0]$ driving risk-consistent loss minimization, Pareto catalog partitioning, and blueprint denoising.

---

## 📊 Comprehensive Head-to-Head Benchmark Results

Evaluating the **Base Model (Vanilla SUPER under Bandwagon Attack $\rho=0.10$)** versus **RRFN-LLM-SUPER** on MovieLens-1M:

| Category | Base Paper Metric | Clean Baseline | Base Model (Attacked) | RRFN-LLM-SUPER (Ours) | Relative Gain (%) | Advantage |
|---|---|---|---|---|---|---|
| **Ranking Accuracy** | `Recall@10` ($\uparrow$) | `0.4910` | `0.4210` | **`0.4820`** | `+14.5%` | 🟢 **WIN (+)** |
| | `nDCG@10` ($\uparrow$) | `0.3840` | `0.3250` | **`0.3780`** | `+16.3%` | 🟢 **WIN (+)** |
| **Popularity Calibration** | `RMSE-PC` ($\downarrow$) | `0.0780` | `0.1420` | **`0.0820`** | `+42.3%` | 🟢 **WIN (42% Lower Error)** |
| | `MRMC` ($\downarrow$) | `0.0910` | `0.1650` | **`0.0940`** | `+43.0%` | 🟢 **WIN (Lower Miscalibration)** |
| **Fairness & Discovery** | `APLT@10 (%)` ($\uparrow$) | `32.0%` | `24.5%` | **`32.4%`** | `+32.2%` | 🟢 **WIN (+)** |
| | `LTC@10 (%)` ($\uparrow$) | `41.2%` | `28.6%` | **`40.8%`** | `+42.7%` | 🟢 **WIN (+)** |
| | `Entropy` ($\uparrow$) | `0.8420` | `0.6810` | **`0.8350`** | `+22.6%` | 🟢 **WIN (+)** |
| | `Novelty (bits)` ($\uparrow$) | `12.85` | `10.40` | **`12.72`** | `+22.3%` | 🟢 **WIN (+)** |
| **Overall Performance** | `GKPI` ($\uparrow$) | `0.6350` | `0.4420` | **`0.6280`** | `+42.1%` | 🟢 **WIN (+)** |
| **Adversarial Robustness** | `ΔGKPI (%)` ($\downarrow$) | `0.0%` | `+30.4%` | **`+1.1%`** | `+96.4%` | 🟢 **WIN (96% Less Degradation)** |
| | `CSS` ($\downarrow$) | `0.0000` | `0.0640` | **`0.0040`** | `+93.8%` | 🟢 **WIN (High Stability)** |
| | `Denoising-F1` ($\uparrow$) | `1.0000` | `0.0000` | **`0.9120`** | `+100.0%` | 🟢 **WIN (High Rejection)** |
| | `Denoising-ROC-AUC` ($\uparrow$) | `1.0000` | `0.5000` | **`0.9450`** | `+89.0%` | 🟢 **WIN (High Discrimination)** |

---

## 📁 Repository Structure

```
├── robust_super/
│   ├── app.py                         # Interactive Streamlit Web Platform
│   ├── configs/
│   │   └── default_config.yaml        # Master Hyperparameter Configuration
│   ├── src/
│   │   ├── data/                      # Loader, Preprocessor, Attack Simulator
│   │   ├── models/                    # NeuMF, LightGCN, VaeCF Neural Backbones
│   │   ├── rrfn/                      # Anchor discovery, Transition Matrix, Risk Loss
│   │   ├── llm_auditor/               # LLM Prompt Builder, Cache, Mock & Live LLM
│   │   ├── bombing_detector/          # Temporal Density, Polarity Skew, Bomb Scorer
│   │   ├── fusion/                    # Multi-View Reliability Fusion
│   │   ├── catalog/                   # Pareto Catalog Partitioning
│   │   ├── super_engine/              # Inclination, Denoised Blueprint, Top-N Merger
│   │   ├── trainer/                   # Decoupled Dual Trainer
│   │   └── evaluation/                # Beyond-Accuracy Metrics, LaTeX Exporter
│   ├── experiments/
│   │   ├── run_baseline.py            # Clean SUPER Baseline Runner
│   │   ├── run_attack.py              # Attacked Vanilla Benchmark
│   │   ├── run_robust_super.py        # RRFN-LLM-SUPER Benchmark
│   │   ├── run_ablations.py           # 7-Variant Systematic Ablation Runner
│   │   └── run_comprehensive_comparison.py # Full Cross-Attack Head-to-Head Runner
│   └── tests/                         # 11-Module Comprehensive Pytest Suite
├── research/                          # Base Research Papers & Foundational Explorations
├── requirements.txt                   # Project Dependencies
├── .gitignore                         # Git Ignore Rules
└── README.md                          # Project Documentation
```

---

## 🚀 Quickstart Guide

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/AdvaitVarhade/RRFN-LLM-SUPER.git
cd RRFN-LLM-SUPER

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Test Suite

```bash
python -m pytest robust_super/tests/ -v
```

### 3. Launch Interactive Web Dashboard

```bash
streamlit run robust_super/app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🖥️ Interactive Web Dashboard Features

The dashboard provides a 6-tab platform:
1. **🌐 Multi-View Signal Fusion**: Donut chart of active weights ($\alpha, \beta, \gamma$), fused reliability weight distribution histogram, and review-bombing sensitivity analysis.
2. **📊 Noise Transition & Denoising Diagnostics**: $T_{\text{final}}$ transition matrix heatmap, ROC-AUC curve, Precision-Recall curve, and training loss convergence curves for dual models ($M_{pop}$ and $M_{tail}$).
3. **🏆 Academic Benchmark & Head-to-Head Comparison**: 13-metric comparative evaluation table, multi-dimensional Radar / Spider plot, grouped bar charts, Attack Budget Robustness Curve ($\rho \in [0.0, 0.25]$), and one-click LaTeX (`.tex`) / CSV exports.
4. **🧪 7-Variant Systematic Ablation Study**: Horizontal bar chart and delta summary evaluating the marginal utility of each component.
5. **🤖 Live LLM Semantic Auditor & Inspector**: Live Google GenAI integration, SQLite cache telemetry, interactive prompt/response walkthrough, and live single-interaction tester.
6. **👤 Live User & Top-10 What-If Sandbox**: Individual user profile viewer, historical ratings with threat flags, side-by-side Top-10 comparison across 3 systems, and an interactive real-time What-If inclination slider.

---

## 📜 Citation & References

```bibtex
@article{super2026,
  title={SUPER: Smart User-centric Popularity Exposure Reduction for Fair and Diverse Recommendations},
  journal={IEEE Access},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
