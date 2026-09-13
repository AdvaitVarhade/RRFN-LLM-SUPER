# RRFN-LLM-SUPER: Robust Risk-Consistent Noise Transition & Lightweight LLM-Audited Popularity Debiasing

## Overview
**RRFN-LLM-SUPER** is an adversarial-resilient, popularity-debiased recommendation framework designed to maintain fairness, diversity, and ranking accuracy even under severe data corruption and coordinated attacks (such as rating-flip noise, bandwagon shilling, and nuke review bombing).

It unifies:
1. **SUPER (IEEE Access 2026)**: Pareto-based decoupled dual modeling ($M_{pop}$ on Head, $M_{tail}$ on Tail) with user-specific popularity inclination calibration and soft blueprint merging.
2. **RRFN Statistical Engine**: 5-class rating probability modeling, anchor-point discovery, noise transition matrix $\hat{T}$ estimation, and Frobenius-regularized $\Delta T$ slack correction.
3. **Lightweight LLM Semantic Auditor**: Asynchronous, pre-filtered persona consistency auditing ($R_{LLM}$) that flags semantic contradictions without adding inference latency.
4. **Spatiotemporal Review-Bombing Detector**: Behavioral burst acceleration $A(i, t)$, polarity concentration $S(i, t)$, and semantic review clustering.
5. **Multi-View Reliability Fusion & Importance Reweighting**: Continuous weights $w_{ui} \in [0.02, 1.0]$ driving risk-consistent loss minimization and robust Pareto catalog partitioning.

---

## Directory Structure
```
robust_super/
├── configs/
│   └── default_config.yaml            # Master configuration parameters
├── data/                              # Data directories (raw / processed)
├── src/
│   ├── data/                          # Loader, Preprocessor, Attack Simulator
│   ├── models/                        # NeuMF, LightGCN, VaeCF Backbones
│   ├── rrfn/                          # Anchor selection, Transition Matrix, Risk Loss
│   ├── llm_auditor/                   # Prompt Builder, Cache, Mock & Live LLM Auditor
│   ├── bombing_detector/              # Temporal Acceleration, Polarity Skew, Scorer
│   ├── fusion/                        # Multi-View Reliability Fusion
│   ├── catalog/                       # Robust Reliability-Weighted Pareto Split
│   ├── super_engine/                  # Robust Inclination, Denoised Blueprint, Top-N Merger
│   ├── trainer/                       # Dual Trainer, Schedulers
│   └── evaluation/                    # Metrics (RMSE-PC, MRMC, GKPI), Robustness, Reporter
├── experiments/
│   ├── run_baseline.py                # Clean SUPER baseline
│   ├── run_attack.py                  # Vanilla SUPER under attack
│   └── run_robust_super.py            # Full RRFN-LLM-SUPER pipeline
└── tests/                             # Comprehensive 11-module pytest suite
```

---

## Quickstart

### 1. Run Unit Tests
```bash
python -m pytest robust_super/tests/ -v
```

### 2. Run Clean SUPER Baseline
```bash
python robust_super/experiments/run_baseline.py
```

### 3. Run Attack Vulnerability Benchmark
```bash
python robust_super/experiments/run_attack.py
```

### 5. Launch the Interactive Dashboard
```bash
python -m streamlit run robust_super/app.py
```
or run `run_dashboard.bat`.

---

## 🌟 Interactive Dashboard Features

The dashboard provides a 6-tab research platform:
1. **🌐 Multi-View Signal Fusion**: Real-time breakdown of Statistical ($R_{RRFN}$), Semantic ($R_{LLM}$), and Behavioral Review-Bombing ($R_{bomb}$) reliability scores, active weight donut chart, and $\omega_1$ (burst) sensitivity analysis.
2. **📊 Noise Transition & Denoising Diagnostics**: Softmax transition matrix heatmap $T_{final} = \text{Softmax}(\hat{T} + \Delta T)$, ROC-AUC curve, Precision-Recall curve, and training loss convergence curves for dual models ($M_{pop}$ and $M_{tail}$).
3. **🏆 Academic Benchmark & Robustness Analysis**:
   - Main **Table II Results Table** comparing Uncalibrated Backbone, Vanilla SUPER (Clean), Vanilla SUPER (Attacked), and RRFN-LLM-SUPER across 7 metrics (`nDCG@10`, `Recall@10`, `RMSE-PC`, `MRMC`, `APLT@10`, `Novelty`, `GKPI`).
   - One-click **"Export LaTeX Table (.tex)"** for copy-pasting directly into research papers.
   - Attack Budget Robustness Curve ($\text{GKPI}$ vs $\rho \in [0.0, 0.25]$) demonstrating slow degradation of our defense under escalating attacks.
   - Grouped attack comparison across 4 modalities (Bandwagon, Nuke Bomb, Random Flip, AGAS).
4. **🧪 7-Variant Systematic Ablation Study**: Interactive horizontal bar chart and delta table isolating the marginal contribution of every component (w/o LLM, w/o RRFN, w/o Bomb, w/o Pareto, w/o Blueprint, w/o Bayesian Shrinkage).
5. **🤖 Live LLM Semantic Auditor & Prompt Inspector**:
   - Google GenAI (`gemini-2.5-flash`) live API integration or lightweight heuristic mock mode.
   - Real-time audit telemetry (Cache hits, live calls, token estimations).
   - Per-interaction prompt and response inspector with exact system prompt, user profile summary, and model rationale.
   - On-demand single interaction audit tester.
6. **👤 Live User & Top-10 Inspector**: Individual user profile viewer, historical training ratings with reliability flags, and Top-10 side-by-side recommendation comparison.

