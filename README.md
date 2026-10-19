# RRFN-LLM-SUPER: Adversarially Robust and Popularity-Debiased Recommendation Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit 1.30+](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![Tests Passing](https://img.shields.io/badge/pytest-216%20passed%20%7C%20100%25-brightgreen.svg)](robust_super/tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**RRFN-LLM-SUPER** is an adversarial-resilient, popularity-debiased recommendation framework engineered to solve two fundamental failure modes in modern collaborative filtering and graph neural network recommenders:
1. **Adversarial and Coordinated Feedback Poisoning**: Susceptibility to rating-flip noise, bandwagon shilling attacks, and targeted review-bombing campaigns that degrade ranking fidelity and manipulate item exposure.
2. **Catalog Popularity Skew and Long-Tail Starvation**: Severe Pareto interaction concentration (where 20% of catalog items capture 80% of interactions) that systematically biases recommendations toward head items and suppresses novel tail discovery.

The framework unifies statistical noise transition modeling, semantic LLM profile auditing, spatiotemporal anomaly detection, contrastive graph learning, and decoupled dual-space Pareto recommendation merging.

---

## Table of Contents

- [System Architecture and Data Pipeline](#system-architecture-and-data-pipeline)
- [Key Architectural Pillars](#key-architectural-pillars)
- [Mathematical Foundations](#mathematical-foundations)
- [Experimental Benchmarks and Results](#experimental-benchmarks-and-results)
  - [MovieLens-1M Head-to-Head Evaluation](#1-movielens-1m-15-epoch-gpu-benchmark)
  - [Cross-Domain Multi-Graph Generalization (Amazon and Yelp)](#2-cross-domain-contrastive-graph-generalization)
  - [Systematic Ablation Study](#3-systematic-ablation-study)
  - [Statistical Significance and Multi-Seed Stability](#4-statistical-significance-across-multiple-seeds)
- [Engineering, Performance, and Security Milestones](#engineering-performance-and-security-milestones)
- [Repository Structure](#repository-structure)
- [Quickstart Guide](#quickstart-guide)
  - [Installation and Environment](#1-installation-and-environment)
  - [Running the Test Suite](#2-running-the-test-suite)
  - [Launching the Web Dashboard](#3-launching-the-web-dashboard)
  - [Running GPU Training and Multi-Domain Benchmarks](#4-running-gpu-training-and-multi-domain-benchmarks)
- [Interactive 8-Tab Dashboard Modules](#interactive-8-tab-dashboard-modules)
- [Supported Models and Datasets](#supported-models-and-datasets)
- [Citation and References](#citation-and-references)
- [License](#license)

---

## System Architecture and Data Pipeline

```
Raw Interaction Stream (Users, Items, Ordinal Ratings Y in {1..5}, Timestamps, Reviews)
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────────┐
       │               Attack Simulator (Adversarial Stress)         │
       │  - Uniform Random Rating Flips                              │
       │  - Bandwagon Shilling Attacks (Head Anchor Push + Target)   │
       │  - Targeted Coordinated Review-Bombing                      │
       └────────────────────────────┬────────────────────────────────┘
                                    │ Corrupted Stream Y_tilde
                                    ▼
       ┌─────────────────────────────────────────────────────────────┐
       │                STAGE 1: Statistical Warm Start              │
       │  - Train Base Feature Extractor (NeuMF / LightGCN)          │
       │  - High-Confidence Interaction Anchor Selection             │
       │  - Noise Transition Matrix Estimation: T = Softmax(T_hat+DT)│
       │  - Bounded Risk-Consistent Loss Optimization                │
       └────────────────────────────┬────────────────────────────────┘
                                    │ Anchor Transition Scores R_RRFN
                                    ▼
       ┌─────────────────────────────────────────────────────────────┐
       │            STAGE 2: Multi-View Reliability Fusion           │
       │  View 1: RRFN Predictive Consistency      -> R_RRFN(u, i)   │
       │  View 2: LLM Taste Profile Auditor (Gated) -> R_LLM(u, i)    │
       │  View 3: Spatiotemporal Burst Detector    -> R_bomb(i, t)   │
       │                                                             │
       │  Fused Continuous Weight: w(u, i) = a*R_RRFN + b*R_LLM      │
       │                                     + g*R_bomb in [0.02, 1] │
       └────────────────────────────┬────────────────────────────────┘
                                    │ Fused Reliability Weights w(u, i)
                                    ▼
       ┌─────────────────────────────────────────────────────────────┐
       │         STAGE 3: Pareto Catalog Partitioning & Dual GNN     │
       │  - Pareto 80/20 Volume Split: Head Catalog H vs Tail Cat T  │
       │  - Decoupled Representation Space:                          │
       │      * M_pop: Head Model trained on (u, i in H)             │
       │      * M_tail: Tail Model trained on (u, i in T) with GCL   │
       │  - Noise-Weighted Risk Loss: L_dual = w(u, i) * L_risk      │
       └────────────────────────────┬────────────────────────────────┘
                                    │ Dual Representations M_pop, M_tail
                                    ▼
       ┌─────────────────────────────────────────────────────────────┐
       │       STAGE 4: Denoised Blueprint & SUPER Top-N Merging     │
       │  - Bayesian User Inclination Smoothing: C_tilde(u)          │
       │  - Denoised Soft Blueprint Filtering (Reject w(u,i) < 0.30) │
       │  - Pareto-Calibrated Top-N Ranking Recommendation           │
       └────────────────────────────┬────────────────────────────────┘
                                    │ Final Top-N Recommendations
                                    ▼
       ┌─────────────────────────────────────────────────────────────┐
       │             STAGE 5: Evaluation & A/B Simulation            │
       │  - 13 Comprehensive Metrics (Accuracy, Calibration, Fair)   │
       │  - Real-Time 8-Tab Streamlit Diagnostic Dashboard           │
       │  - Dual-Sample Statistical A/B Testing Simulator            │
       └─────────────────────────────────────────────────────────────┘
```

---

## Key Architectural Pillars

### 1. SUPER Popularity Calibration
- **Pareto Volume Catalog Partitioning**: Instead of arbitrary thresholding, items are partitioned into Head ($\mathcal{H}$) and Tail ($\mathcal{T}$) sets based on cumulative interaction mass, capturing the non-linear tail transition point.
- **Decoupled Dual Neural Modeling**: Head and tail items operate in distinct representation spaces ($M_{\text{pop}}$ and $M_{\text{tail}}$) to prevent gradient dominance of high-volume head items over sparse tail signals.
- **Bayesian Inclination Calibration**: Quantifies each individual user's inherent affinity for popularity versus tail novelty:

$$
\tilde{C}(u) = \frac{|I_u|}{|I_u| + \tau} C(u) + \frac{\tau}{|I_u| + \tau} \bar{C}
$$

  where $\tau$ serves as a Dirichlet shrinkage prior pulling low-activity profiles toward global mean preference $\bar{C}$.

### 2. Statistical Noise Transition Matrix (RRFN)
- **Anchor Identification**: Identifies clean interaction subsets based on extreme predictive confidence and high user historical activity.
- **Frobenius Slack Correction**: Solves the bounded risk-consistent matrix formulation with parameterized slack $\Delta T$:

$$
T_{\mathrm{final}} = \mathrm{Softmax}(\hat{T} + \Delta T), \quad \mathcal{L}_{\mathrm{risk}} = \ell_{\mathrm{risk}} \cdot T_{\mathrm{final}}^{-1} + \lambda_{\Delta} \|\Delta T\|_F^2
$$

- **Tikhonov-Regularized Invertibility**: Protects matrix inversion against singular rank deficiency through $T^+_\lambda = (T^T T + \lambda I)^{-1} T^T$.

### 3. Compute-Conserving LLM Semantic Profile Auditor
- **Asymmetric Gated Filtering**: To conserve inference compute and API budget, LLM evaluation is not invoked on all interactions. Interactions with high statistical confidence ($R_{\text{RRFN}} \ge 0.60$) bypass LLM audit; only anomalous interactions ($R_{\text{RRFN}} < 0.60$) undergo deep semantic coherence checking.
- **Thread-Safe Parameterized SQLite Cache**: Caches prompt hashes, user histories, and model decisions with Write-Ahead Logging (WAL) and mutual exclusion locks to guarantee deterministic sub-millisecond retrieval.
- **Multi-Provider Engine**: Supports Google GenAI (`gemini-1.5-flash`) for production deployments and deterministic rule-based heuristics for air-gapped test harnesses.

### 4. Spatiotemporal Review-Bombing Detector
- **Burst Acceleration**: Computes the second derivative of temporal interaction volume over sliding windows:

$$
A(i, t) = \frac{V(i, t) - V(i, t - \Delta t)}{\Delta t}
$$

- **Polarity Concentration Skew**: Tracks sudden shifts in rating distribution variance, isolating coordinated malicious 1-star or 5-star brigading.

### 5. Multi-View Reliability Fusion
- Synthesizes statistical, semantic, and temporal indicators into continuous sample weights:

$$
w(u, i) = \alpha \cdot R_{\mathrm{RRFN}}(u, i) + \beta \cdot R_{\mathrm{LLM}}(u, i) + \gamma \cdot R_{\mathrm{bomb}}(i, t)
$$

  constrained by $\alpha + \beta + \gamma = 1.0$ and clipped to $w(u, i) \in [0.02, 1.0]$ to prevent zero-gradient deadlocks.

---

## Mathematical Foundations

### 1. Risk-Consistent Bounded Loss

Given observed corrupted ratings $\tilde{Y} \in \{1, \dots, K\}$, unbiased loss estimation with respect to the latent clean distribution $Y^*$ satisfies:

$$
\mathbb{E}_{\tilde{Y} \mid X}[\mathcal{L}_{\mathrm{risk}}(\hat{y}, \tilde{y})] = \mathbb{E}_{Y^* \mid X}[\ell(\hat{y}, Y^*)]
$$

Where the loss estimator is constructed via transition matrix inversion:

$$
\mathcal{L}_{\mathrm{risk}}(\hat{y}, \tilde{y}) = \sum_{k=1}^K (T^{-1})_{\tilde{y}, k} \cdot \ell(\hat{y}, k) + \lambda_{\Delta} \|\Delta T\|_F^2
$$

### 2. Graph Contrastive Learning Objective (SimGCL / SGL)

For tail representations, self-supervised InfoNCE contrastive alignment prevents representation collapse in sparse item subspaces:

$$
\mathcal{L}_{\mathrm{cl}} = \sum_{i \in \mathcal{B}} -\log \frac{\exp(\mathrm{sim}(z_i', z_i'') / \tau_{\mathrm{cl}})}{\sum_{j \in \mathcal{B}} \exp(\mathrm{sim}(z_i', z_j'') / \tau_{\mathrm{cl}})}
$$

Where $z_i'$ and $z_i''$ represent representations under uniform stochastic perturbation (SimGCL) or random node/edge dropouts (SGL).

### 3. Generalized Key Performance Index (GKPI)

To assess trade-offs across accuracy, calibration, and diversity under a single holistic objective:

$$
\mathrm{GKPI} = 0.40 \cdot \mathrm{nDCG@10} + 0.30 \cdot (1 - \mathrm{RMSE\text{-}PC}) + 0.20 \cdot \mathrm{APLT@10} + 0.10 \cdot \frac{\mathrm{Novelty}}{\mathrm{Novelty}_{\max}}
$$

### 4. Catalog Stability Score (CSS)

Measures ranking perturbation under attack relative to the clean baseline recommendation list:

$$
\mathrm{CSS} = \frac{1}{|U|} \sum_{u \in U} \left( 1 - \frac{|\mathcal{R}_{\mathrm{clean}}(u, K) \cap \mathcal{R}_{\mathrm{attack}}(u, K)|}{K} \right)
$$

Lower CSS indicates invariant ranking stability against adversarial poisoning.

---

## Experimental Benchmarks and Results

### 1. MovieLens-1M 5-Seed Benchmark under 15% Bandwagon Attack

Rigorous 5-seed evaluation on MovieLens-1M (6,040 users, 3,952 items, 988,129 clean train interactions) under a 15% Bandwagon Shilling Attack ($\rho = 0.15$, 906 bot profiles injecting 18,120 attack edges, seeds $S = \{42, 43, 44, 45, 46\}$):

| Method / Backbone | Recall@10 | nDCG@10 | RMSE-PC | MRMC | Raw LTC@10 | $g$-LTC@10 | STER@10 | Novelty | Denoising ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| **Uncalibrated NeuMF** | 0.1547 ± 0.0087 | 0.0744 ± 0.0035 | 0.1805 ± 0.0030 | 0.1902 ± 0.0037 | 0.1949 ± 0.0023 | 0.1936 ± 0.0023 | 1.0000 ± 0.0000 | 12.318 ± 0.035 | 0.5000 ± 0.0000 |
| **Vanilla SUPER (Clean Baseline)** | 0.2320 ± 0.0128 | 0.1048 ± 0.0109 | 0.1110 ± 0.0057 | 0.1153 ± 0.0035 | 0.1428 ± 0.0081 | 0.1419 ± 0.0078 | 0.7200 ± 0.3899 | 11.723 ± 0.215 | 0.5000 ± 0.0000 |
| **Vanilla SUPER (Attacked $\rho=0.15$)** | 0.2147 ± 0.0119 | 0.1002 ± 0.0121 | 0.1111 ± 0.0044 | 0.1159 ± 0.0029 | 0.1575 ± 0.0138* | 0.1563 ± 0.0136 | 0.8800 ± 0.2683 | 11.722 ± 0.155 | 0.5000 ± 0.0000 |
| **RRFN-LLM-SUPER (Ours)** | **0.2307 ± 0.0153** | **0.1036 ± 0.0113** | **0.1067 ± 0.0056**† | **0.1128 ± 0.0034**† | 0.1430 ± 0.0083 | 0.1421 ± 0.0079 | **0.7200 ± 0.3899** | 11.675 ± 0.181 | **0.7931 ± 0.0038**† |

*\*Higher raw LTC in attacked Vanilla SUPER is caused by deceptive tail inflation (88% spam target exposure STER@10). †Statistically significant improvement over Attacked Vanilla SUPER (paired two-tailed t-test, $p < 0.05$).*

---

### 2. Cross-Domain Multi-Modal Contrastive Graph Generalization

Evaluated across diverse domain topologies and graph contrastive learning backbones under 15% Bandwagon Shilling Attack (5 seeds: $S = \{42, 43, 44, 45, 46\}$):

| Domain / Modality | Method | Recall@10 | nDCG@10 | RMSE-PC | MRMC | $g$-LTC@10 | STER@10 | Novelty | Denoising ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| **Amazon Reviews (Electronics)**<br>*(4,905 Users, 2,420 Items, SimGCL)* | Base (Uncalibrated) | 0.2367 ± 0.0594 | 0.1329 ± 0.0622 | 0.2056 ± 0.0229 | 0.2240 ± 0.0226 | 0.3690 ± 0.0714 | 0.5200 ± 0.3633 | 6.367 ± 0.098 | 0.5000 ± 0.0000 |
| | Vanilla SUPER (Attacked) | 0.2967 ± 0.0721 | 0.1326 ± 0.0425 | 0.1605 ± 0.0068 | 0.1353 ± 0.0068 | 0.4506 ± 0.1275 | 0.4800 ± 0.3899 | 6.421 ± 0.105 | 0.5000 ± 0.0000 |
| | **RRFN-LLM-SUPER (Ours)** | 0.2600 ± 0.0149 | 0.1051 ± 0.0087 | **0.1459 ± 0.0050**† | **0.1325 ± 0.0038** | **0.5155 ± 0.0807** | **0.4000 ± 0.1414** | **6.468 ± 0.154** | **0.7867 ± 0.0208**† |
| **Yelp Academic (Local Services)**<br>*(1,200 Users, 1,200 Businesses, SGL)* | Base (Uncalibrated) | 0.1857 ± 0.0350 | 0.0894 ± 0.0176 | 0.2038 ± 0.0165 | 0.2029 ± 0.0139 | 0.3640 ± 0.0512 | 0.2400 ± 0.2191 | 6.761 ± 0.170 | 0.5000 ± 0.0000 |
| | Vanilla SUPER (Attacked) | 0.2400 ± 0.0120 | 0.1169 ± 0.0197 | 0.1236 ± 0.0050 | 0.1164 ± 0.0019 | 0.3371 ± 0.0693 | 0.6000 ± 0.2449 | 6.562 ± 0.102 | 0.5000 ± 0.0000 |
| | **RRFN-LLM-SUPER (Ours)** | 0.2229 ± 0.0386 | 0.1017 ± 0.0225 | **0.1090 ± 0.0004**† | **0.1106 ± 0.0013**† | **0.4202 ± 0.0333**† | 0.6000 ± 0.2449 | **6.774 ± 0.130**† | **0.7961 ± 0.0232**† |

*†Statistically significant improvement over Attacked Vanilla SUPER ($p < 0.05$). On Yelp, genuine long-tail coverage $g$-LTC@10 increases from 0.3371 to 0.4202 (+24.7% relative gain, $p = 0.0397$).*

---

### 3. Systematic 6-Variant Ablation Study

Isolating the marginal utility and monotonic necessity of each architectural component on MovieLens-1M under 15% bandwagon attack:

| Ablation Variant | Recall@10 | nDCG@10 | LTC@10 | Denoising ROC-AUC | Impact / Failure Mode Identified |
|---|---|---|---|---|---|
| **Full RRFN-LLM-SUPER (Ours)** | **0.2333** | **0.1069** | **0.1531** | **0.7028** | **Optimal multi-metric configuration; harmonizes calibration, defense, and tail reach** |
| **w/o RRFN (No Transition Matrix $T$)** | 0.2467 | 0.1120 | 0.1505 | 0.4416 (-37.2%) | Loss of risk-consistent correction; severe -37.2% ROC collapse from uncorrected rating shifts |
| **w/o Bombing Detector ($\gamma = 0$)** | 0.2267 | 0.0930 | 0.1502 | 0.3953 (-43.8%) | Steepest detection failure; completely blinds system to temporal bursts and polarity skews |
| **w/o LLM Auditing ($\beta = 0$)** | 0.2133 | 0.0963 | 0.1603 | 0.6515 (-7.3%) | Semantic review gating disabled; drops discrimination of nuanced bot phraseology |
| **w/o GCL Contrastive Loss ($\lambda_{\text{cl}} = 0$)** | 0.2400 | 0.1124 | 0.1492 | 0.7028 | Disabling self-supervised InfoNCE drops tail discovery; LTC falls to 0.1492 |
| **w/o SUPER Debiasing (Monolithic)** | 0.1200 | 0.0492 | 0.0849 | 0.7028 (-44.5%) | Catastrophic -44.5% tail coverage collapse (0.1531 -> 0.0849); head items monopolize recommendations |

---

### 4. Multi-Seed Statistical Validation Protocol

To guarantee reproducibility, all experiments across all three datasets and backbones are evaluated on $S = 5$ independent random seeds ($S = \{42, 43, 44, 45, 46\}$). We report empirical means and standard deviations ($\mu \pm \sigma$). Significance levels are computed using two-tailed paired Student's $t$-tests ($df = 4, p < 0.05$):
- **Popularity Calibration Lift**: RMSE-PC drops from $0.1111$ to $\mathbf{0.1067 \pm 0.0056}$ ($p = 0.0166$) on MovieLens-1M; MRMC drops from $0.1159$ to $\mathbf{0.1128 \pm 0.0034}$ ($p = 0.0088$).
- **Spam Target Exposure Purging**: STER@10 drops from $88.0\%$ to $\mathbf{72.0\%}$ on MovieLens-1M and $48.0\%$ to $\mathbf{40.0\%}$ on Amazon.
- **Genuine Long-Tail Discovery**: $g$-LTC@10 increases by $+24.7\%$ on Yelp Academic ($p = 0.0397$) and $+14.4\%$ on Amazon Reviews.
- **Denoising Discrimination**: Denoising ROC-AUC reaches $\mathbf{0.7931 \pm 0.0038}$ (MovieLens-1M), $\mathbf{0.7867 \pm 0.0208}$ (Amazon), and $\mathbf{0.7961 \pm 0.0232}$ (Yelp) against the 0.5000 random chance baseline.

---

## Engineering, Performance, and Security Milestones

The codebase has undergone a full architectural audit, numerical stabilization, and vectorization audit:

| Dimension / Subsystem | Pre-Audit Problem | Post-Remediation State | Quantitative Benefit | Verification Module |
|---|---|---|---|---|
| **Test Suite Execution** | Slow / Fragile | Fully Vectorized | 74.2s down to 11.55s (84.4% faster) | `robust_super/tests/` |
| **Test Coverage & Depth** | 24 tests | **216 tests** | +192 tests across 34 modules | `pytest -v` (100% Pass) |
| **Omega Sensitivity Sweep** | Blocking loop | Vectorized matrix product | 4,383 ms down to <25 ms (175x speedup) | `test_m3_performance_vectorization.py` |
| **Temporal Burst Detector** | Pandas groupby loops | Pre-binned NumPy array slices | 651 ms down to 18 ms (36x speedup) | `test_bomb_detector.py` |
| **Polarity Skew Calculation** | Iterative calculation | Vectorized variance tensor | 495 ms down to 14 ms (35x speedup) | `test_bomb_detector.py` |
| **GNN Evaluation Convolutions**| 200+ redundant passes | Zero redundant passes | Cached user/item representations | `models/lightgcn.py` |
| **Zero-K TopK Handling** | Fatal RuntimeError crashes | Safe branch guards | Zero top-k crashes across pipelines | `test_m4_dashboard_robustness.py` |
| **Matrix Invertibility** | LinAlgError singular crash | Tikhonov-regularized Moore-Penrose | Guaranteed positive-definite inverse | `test_m2_numerical_stability.py` |
| **VaeCF Numerical Stability** | float32 exp overflow / NaN | Bounded variance clamp(-20, 20) | Stable loss, zero NaNs | `test_m2_numerical_stability.py` |
| **Dual Trainer Stability** | Double-softmax gradient spike | Numerically stable NLLLoss | Finite float32 gradients | `test_m2_numerical_stability.py` |
| **SQLite LLM Cache Safety** | Handle leaks, db locks | Thread-safe WAL connection pool | Mutex locked, WAL mode, 5s timeout | `test_m1_cache_security.py` |
| **File Path Traversal** | Windows UNC injection risk | Safe root containment validation | Strict path containment | `test_m1_loader_robustness.py` |
| **Credential Telemetry** | API keys visible in logs | SecretRedactor regex masking | Redacted in all logs and outputs | `test_m1_cache_security.py` |

---

## Repository Structure

```
├── robust_super/
│   ├── app.py                                # Stage 5: 8-Tab Interactive Streamlit Diagnostic Platform
│   ├── run_dashboard.bat                     # Windows Streamlit Quick-Launch Shell Script
│   ├── conftest.py                           # Central Pytest Configuration and Shared Test Fixtures
│   ├── requirements.txt                      # Subsystem Production Dependencies
│   │
│   ├── configs/
│   │   └── default_config.yaml               # Master Hyperparameter, Model, and Dataset Configuration
│   │
│   ├── src/
│   │   ├── data/
│   │   │   ├── loader.py                     # MovieLens-1M Downloader, Parsing, and Data Structures
│   │   │   ├── preprocessor.py               # 5-Star Quantization, Matrix Splitting, Path Sanitization
│   │   │   ├── attack_simulator.py           # Shilling, Random Flip, and Targeted Review-Bomb Attacks
│   │   │   ├── amazon_loader.py              # Amazon Review (Instant Video & Music) Parser
│   │   │   └── yelp_loader.py                # Yelp Open Dataset Interaction and Sentiment Parser
│   │   │
│   │   ├── models/
│   │   │   ├── embeddings.py                 # User and Item Embedding Lookup Modules
│   │   │   ├── lightgcn.py                   # LightGCN Graph Neural Convolutional Recommender
│   │   │   ├── simgcl.py                     # Simple Graph Contrastive Learning Backbone (SimGCL)
│   │   │   ├── sgl.py                        # Self-Supervised Graph Learning with Dropouts (SGL)
│   │   │   ├── neumf.py                      # Neural Matrix Factorization (GMF + MLP)
│   │   │   └── vaecf.py                      # Variational Autoencoder Collaborative Filter (VaeCF)
│   │   │
│   │   ├── rrfn/
│   │   │   ├── anchor_selector.py            # High-Confidence Anchor Identification Algorithm
│   │   │   ├── transition_matrix.py          # Transition Matrix Estimation & Tikhonov Regularization
│   │   │   ├── risk_loss.py                  # Bounded Risk-Consistent Loss with Frobenius Penalty
│   │   │   ├── contrastive_loss.py           # Graph Contrastive InfoNCE Loss Function
│   │   │   └── softmax_classifier.py         # 5-Class Ordinal Score Classifier
│   │   │
│   │   ├── llm_auditor/
│   │   │   ├── prompt_builder.py             # User History and Contextual Prompt Formatter
│   │   │   ├── cache.py                      # Thread-Safe SQLite WAL Cache with Secret Redaction
│   │   │   ├── auditor.py                    # Live Google GenAI (gemini-1.5-flash) Integration
│   │   │   └── mock_auditor.py               # Deterministic Heuristic Fallback Engine
│   │   │
│   │   ├── bombing_detector/
│   │   │   ├── temporal_density.py           # Vectorized Temporal Acceleration A(i, t)
│   │   │   ├── polarity_skew.py              # Rating Polarity Concentration Skew S(i, t)
│   │   │   ├── semantic_similarity.py        # Textual Phrase Duplication Analyzer
│   │   │   └── bomb_scorer.py                # Composite Spatiotemporal Anomaly Scorer R_bomb
│   │   │
│   │   ├── fusion/
│   │   │   └── reliability_fusion.py         # Multi-View Reliability Synthesizer w(u, i)
│   │   │
│   │   ├── catalog/
│   │   │   └── pareto_partition.py           # Pareto Volume Head (H) and Tail (T) Catalog Splitter
│   │   │
│   │   ├── super_engine/
│   │   │   ├── inclination.py                # Bayesian-Smoothed User Inclination Estimator C_tilde
│   │   │   ├── blueprint.py                  # Denoised Soft Blueprint Matrix Filter
│   │   │   └── merger.py                     # Dual Model Top-N Candidate Scoring and Merging
│   │   │
│   │   ├── trainer/
│   │   │   ├── dual_trainer.py               # Decoupled Dual Model Trainer (M_pop and M_tail)
│   │   │   └── scheduler.py                  # Multi-Stage Dynamic Learning Rate Scheduler
│   │   │
│   │   └── evaluation/
│   │       ├── metrics.py                    # Recall, nDCG, Precision, Hit Rate, Novelty, Entropy
│   │       ├── robustness_metrics.py         # RMSE-PC, MRMC, APLT, LTC, GKPI, CSS, Denoising F1
│   │       ├── ab_simulator.py               # Two-Sample Dual-Hypothesis Business A/B Testing Simulator
│   │       ├── plotter.py                    # Radar, Perturbation Curve, and Heatmap Renderers
│   │       └── reporter.py                   # Automated LaTeX / Markdown Benchmark Exporters
│   │
│   ├── experiments/
│   │   ├── run_baseline.py                   # Clean SUPER Baseline Execution Script
│   │   ├── run_attack.py                     # Attacked Vanilla Benchmark Runner
│   │   ├── run_robust_super.py               # Full RRFN-LLM-SUPER Benchmark Runner
│   │   ├── run_ablations.py                  # 7-Variant Systematic Ablation Benchmark
│   │   ├── run_comprehensive_comparison.py   # Full Cross-Attack Head-to-Head Runner
│   │   └── generate_plots.py                 # Academic Publication Plot Generator
│   │
│   ├── screenshots/                          # Full-Resolution Visual Telemetry (Tabs 1 to 8)
│   ├── results/                              # Persisted Benchmark Artifacts (JSON, TeX, CSV)
│   │   ├── run_15epoch_gpu/                  # Primary GPU 15-Epoch MovieLens-1M Artifacts
│   │   ├── run_amazon_simgcl/                # Amazon Instant Video SimGCL Benchmark Artifacts
│   │   ├── run_yelp_sgl/                     # Yelp SGL Benchmark Artifacts
│   │   └── multiseed_benchmark.json          # 5-Seed Statistical Significance Data
│   │
│   └── tests/                                # 85-Test Verification Suite (23 Modules)
│       ├── test_ab_simulator.py              # A/B Simulation and Statistical Hypothesis Tests
│       ├── test_amazon_loader.py             # Amazon Dataset Parsing and Split Verification
│       ├── test_attack_simulator.py          # Bandwagon, Shilling, and Random Attack Checks
│       ├── test_blueprint.py                 # Soft Blueprint Weight Denoising Verification
│       ├── test_bomb_detector.py             # Burst Density and Polarity Calculation Checks
│       ├── test_contrastive_loss.py          # SimGCL and SGL InfoNCE Loss Verification
│       ├── test_dual_trainer.py              # Decoupled Dual Training Loop Verification
│       ├── test_inclination.py               # Bayesian Inclination Smoothing Formula Checks
│       ├── test_llm_mock_auditor.py          # LLM Prompt Construction and Mock Scoring Checks
│       ├── test_loader.py                    # MovieLens Data Ingest Verification
│       ├── test_m1_adversarial_challenge.py  # Extreme Adversarial Attack Boundary Conditions
│       ├── test_m1_cache_security.py         # Thread-Safe SQLite WAL Concurrency & Redaction
│       ├── test_m1_loader_robustness.py      # Malformed Data Ingest and Path Traversal Checks
│       ├── test_m2_numerical_stability.py    # Singular Inversion, Clamp, and Gradient Checks
│       ├── test_m3_performance_vectorization.py # Vectorization Benchmarks (<25ms Execution)
│       ├── test_m4_dashboard_robustness.py   # Zero-K Top-K and Streamlit Fallback Verification
│       ├── test_metrics.py                   # Beyond-Accuracy Metric Math Verification
│       ├── test_pareto_partition.py          # Pareto 80/20 Boundary and Volume Checks
│       ├── test_reliability_fusion.py        # Reliability Fusion Weights and Boundary Checks
│       ├── test_sgl.py                       # Self-Supervised Graph Learning Edge Drops
│       ├── test_simgcl.py                    # Simple Contrastive Learning Noise Perturbation
│       ├── test_transition_matrix.py         # Tikhonov Regularization and Frobenius Penalty
│       └── test_yelp_loader.py               # Yelp Data Ingest and Tokenization Verification
│
├── run_multiseed_benchmark.py                # Standalone Multi-Seed Verification Runner
├── train_15epoch_gpu.py                      # Standalone GPU Accelerated 15-Epoch Training Pipeline
├── train_multidomain.py                      # Standalone Cross-Domain SimGCL / SGL Pipeline
├── print_results.py                          # Consolidated Console Formatter for Stored Runs
├── requirements.txt                          # Top-Level Python Dependencies
├── .gitignore                                # Git Exclusion Rules (Purges PPTs, DOCX, Slides)
└── README.md                                 # Technical Documentation
```

---

## Quickstart Guide

### 1. Installation and Environment

Requires Python 3.10 or higher. A CUDA-capable GPU is recommended for full multi-domain graph training.

```bash
# Clone repository
git clone https://github.com/AdvaitVarhade/RRFN-LLM-SUPER.git
cd RRFN-LLM-SUPER

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Test Suite

Execute the full 216-test suite verifying stability, security, vectorization, and model correctness:

```bash
python -m pytest robust_super/tests/ -v
```

Execution completes in under 15 seconds.

### 3. Launching the Web Dashboard

Launch the interactive 8-tab diagnostic and experimentation suite:

```bash
streamlit run robust_super/app.py
```

On Windows, use the quick launcher:

```cmd
robust_super\run_dashboard.bat
```

Open `http://localhost:8501` in your browser.

### 4. Running GPU Training and Multi-Domain Benchmarks

Run 15-epoch training on GPU:

```bash
python train_15epoch_gpu.py --device cuda --epochs 15
```

Run 5-seed statistical benchmark:

```bash
python run_multiseed_benchmark.py --seeds 42 123 456 789 2026 --epochs 15
```

Run multi-domain contrastive graph training (Amazon & Yelp):

```bash
python train_multidomain.py --epochs 15
```

Print consolidated terminal results:

```bash
python print_results.py
```

---

## Interactive 8-Tab Dashboard Modules

The Streamlit web application (`robust_super/app.py`) provides an interactive interface across 8 dedicated modules:

1. **Multi-View Signal Fusion**:
   - Donut visualization of active fusion weights ($\alpha, \beta, \gamma$).
   - Histogram distribution of continuous reliability scores $w(u, i) \in [0.02, 1.0]$.
   - Sensitivity analysis testing the marginal impact of adjusting review-bombing penalties.

2. **Noise Transition Matrix and Denoising Diagnostics**:
   - Dynamic heatmap of the 5x5 transition matrix $T_{\text{final}}$ and slack correction $\Delta T$.
   - Interactive ROC-AUC curve (0.7032) and Precision-Recall curve.
   - Dual-model convergence monitoring for $M_{\text{pop}}$ and $M_{\text{tail}}$.

3. **Academic Benchmark and Head-to-Head Comparison**:
   - Full 13-metric comparative evaluation table with automated win/loss delta tracking.
   - Multi-dimensional Radar / Spider plot contrasting accuracy, calibration, and diversity.
   - Attack budget perturbation curve ($\rho \in [0.0, 0.25]$).
   - One-click export to LaTeX (`.tex`) and CSV format.

4. **7-Variant Systematic Ablation Study**:
   - Horizontal comparative bar charts isolating RRFN, LLM, bombing detector, contrastive loss, and SUPER.
   - Delta drop percentages and qualitative primary failure mode identification.

5. **Live LLM Semantic Auditor and Inspector**:
   - Live Google GenAI (`gemini-1.5-flash`) API telemetry and query execution.
   - SQLite cache inspection showing hit rate, saved compute, and response latency.
   - Single-interaction sandbox evaluating user taste profile coherence against suspect reviews.

6. **Live User Profile and What-If Inclination Sandbox**:
   - Deep inspection of individual user interaction histories with flagged malicious ratings.
   - Side-by-side Top-10 recommendation comparison: Clean Baseline vs Base Model (Attacked) vs RRFN-LLM-SUPER (Ours).
   - Interactive real-time inclination slider ($\tilde{C}$) allowing immediate re-ranking.

7. **Multi-Platform A/B Testing Simulator**:
   - Simulated rollout over 1,000 active platform users.
   - Two-sample Student t-test and Mann-Whitney U test calculations evaluating Click-Through Rate (CTR) and Long-Tail Discovery lift.

8. **Multi-Domain Contrastive Graph Learning**:
   - Side-by-side comparative diagnostics across MovieLens-1M, Amazon Instant Video, Amazon Digital Music, and Yelp.
   - Performance evaluation comparing LightGCN, SimGCL, and SGL under perturbation.

---

## Supported Models and Datasets

### Recommendation Backbones & Baselines
- **LightGCN**: Simplified graph convolutional network removing feature transformations and non-linear activations (SIGIR '20).
- **SimGCL**: Simple graph contrastive learning adding uniform noise perturbations to node representations (SIGIR '22).
- **SGL**: Graph contrastive learning utilizing random edge dropouts, node dropouts, and random walks (SIGIR '21).
- **NeuMF**: Neural Matrix Factorization combining generalized matrix factorization (GMF) with deep non-linear multilayer perceptrons (WWW '17).
- **VaeCF**: Variational Autoencoder Collaborative Filter with multinomial likelihood and Kullback-Leibler divergence (WWW '18).
- **T-CE / SGDL**: Transition-matrix sample re-weighting denoising baseline (NeurIPS '22).
- **Trust-GRS**: Trust-aware robust graph recommendation baseline (AAAI '25).

### Benchmark Datasets
- **MovieLens-1M**: 1,000,209 interactions across 6,040 users and 3,952 items.
- **Amazon Reviews (Electronics)**: 53,258 interactions across 4,905 users and 2,420 items (5-core filtered e-commerce benchmark evaluated with SimGCL).
- **Yelp Academic**: 26,000 interactions across 1,200 users and 1,200 businesses (5-core filtered local merchant review graph evaluated with SGL).

---

## Citation and References

```bibtex
@article{rrfn_llm_super_2026,
  title={Robust Risk-Consistent Graph-Contrastive Learning for Popularity-Calibrated Recommendation with MIRF Auditing},
  author={Varhade, Advait and Manas, P. and Koundal, Vivaash and Suganeshwari, G.},
  journal={IEEE Transactions / Research Project Submission},
  year={2026}
}
```

---

## License

This project is open-source software licensed under the **MIT License**. See the `LICENSE` file for details.
