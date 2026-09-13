# TEST_INFRA: 4-Tier Opaque-Box Test Architecture for FedSUPER-LLM Dynamic Merge

## 1. Overview & Architecture

This document defines the comprehensive test infrastructure and 4-tier opaque-box verification methodology for the **FedSUPER-LLM Dynamic Blueprint Merge** research framework.

The primary mission of the test suite (`tests/test_h9_advanced_merge.py`) is to rigorously validate the mathematical invariants, algorithmic correctness, boundary resilience, cross-feature composability, and real-world workload acceptance criteria ($\text{Recall@20} > 0.037$ and $\text{Rmse-PC} \le 0.056$) across all merge and calibration algorithms.

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                     4-Tier Opaque-Box Test Suite Architecture                     │
├───────────────────────────────────────────────────────────────────────────────────┤
│ Tier 1: Feature Coverage (Unit & Algorithmic Correctness, >= 5 cases / feature)   │
│   ├─ 1.1 Stochastic User Quota (Sum, Unbiased Expectation, Variance Bounds)       │
│   ├─ 1.2 User-Adaptive Alpha Partition (Activity Scaling, Boundary Invariance)    │
│   ├─ 1.3 Confidence-Elastic Merge (Score Margins, Slack Bounding, Positive Mask)  │
│   ├─ 1.4 Dynamic Probabilistic Quota Merge (All Modes: Stochastic, Adaptive, etc.)│
│   ├─ 1.5 Calibrated Score Fusion (Intra-Pool Z-Score Standardization, Lambda Blend)│
│   └─ 1.6 Popularity Calibration & Metric Evaluation (Rmse-PC, MRMC, GKPI)         │
├───────────────────────────────────────────────────────────────────────────────────┤
│ Tier 2: Boundary & Corner Cases (>= 5 cases / feature)                            │
│   ├─ Slate Sizes: N = 1, N = 2, N = 20, N = 50, N = 100                           │
│   ├─ Extremes: Pop_u = 0.0 (Pure Tail) and Pop_u = 1.0 (Pure Head)                │
│   ├─ Cold-Start: Zero interaction histories (|C_u| = 0), All-item interacted users│
│   ├─ Catalog Degeneracy: Single-item head pool (|H| = 1), Tiny tail pool          │
│   └─ Numerical Extremes: Zero variance, Inf/-Inf, Huge score gaps (+/- 1e6)       │
├───────────────────────────────────────────────────────────────────────────────────┤
│ Tier 3: Cross-Feature Interactions & Combinations                                 │
│   ├─ Stochastic Quota + User-Adaptive Alpha Modulation                            │
│   ├─ SBERT Score Fusion (λ=0.7) + Confidence-Elastic Interleaving                 │
│   ├─ Dynamic Merge + Full-Rank Metric Evaluation Pipeline                         │
│   ├─ Multi-Tier Partition (Head/Torso/Tail) + Calibrated Fusion                   │
│   └─ Pareto Partition + Adaptive Alpha + Stochastic Quota + Train-Pos Masking     │
├───────────────────────────────────────────────────────────────────────────────────┤
│ Tier 4: Real-World Workload Scenarios & Acceptance Criteria                       │
│   ├─ Full ML-1M Scale Workload (6,040 users, 3,533 items, α=0.20, K=20)          │
│   ├─ Acceptance Verification: Recall@20 > 0.0370 and Rmse-PC <= 0.0560            │
│   └─ High-Concurrency Execution & Latency Benchmark                              │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 4-Tier Test Coverage Matrix

### Tier 1: Feature Coverage (Contract & Mathematical Invariants)

| Feature | Primary Function | Core Assertions & Invariants | Min Cases |
|---|---|---|---|
| **F1: Stochastic Quota** | `stochastic_user_quota` | (1) $N_{\text{pop}} + N_{\text{tail}} \equiv N$, (2) $0 \le N_{\text{pop}}, N_{\text{tail}} \le N$, (3) $\left\|\mathbb{E}[N_{\text{pop}}] - N \cdot \text{Pop}_u\right\| < 10^{-2}$, (4) $\left|\frac{N_{\text{pop}}}{N} - \text{Pop}_u\right| \le \frac{1}{N} + \epsilon$, (5) Determinism under `rng`. | $\ge 5$ |
| **F2: Adaptive Alpha** | `user_adaptive_alpha_partition` | (1) Output shape $(n_u,)$, (2) Values in $[0.0, 1.0]$, (3) $\beta=0 \implies \text{Pop}_u$, (4) Active vs sparse scaling $w_u = \frac{\log(1+|C_u|)}{\max \log(1+|C_v|)}$, (5) Zero interaction fallback. | $\ge 5$ |
| **F3: Confidence Elastic**| `confidence_elastic_merge` | (1) Output shape $(n_u, N)$, (2) No duplicates per user slate, (3) Zero train positive leakage ($M_{u, i} = 0$), (4) Margin $\Delta > \theta$ triggers dynamic quota swap, (5) Depleted pool fallback. | $\ge 5$ |
| **F4: Dynamic Merge** | `dynamic_probabilistic_quota_merge`| (1) Mode `'stochastic'`, (2) Mode `'adaptive_alpha'`, (3) Mode `'confidence_elastic'`, (4) Mode `'hybrid'`, (5) Mode `'static'` backwards compatibility, (6) Seed invariance. | $\ge 6$ |
| **F5: Score Fusion** | `calibrated_score_fusion` | (1) $\lambda=0.0$ matches fed, (2) $\lambda=1.0$ matches LLM, (3) Intra-pool $\mu=0, \sigma=1$ standardization, (4) Zero-variance handling, (5) Non-pool score isolation. | $\ge 5$ |
| **F6: Calibration Eval** | `evaluate_calibration` / `full_rank_eval` | (1) Hand-calculated Rmse-PC verification, (2) MRMC ranking monotonicity, (3) GKPI harmonic mean validation, (4) Full-rank dict keys completeness, (5) Bound $\le 0.056$. | $\ge 5$ |

---

### Tier 2: Boundary & Corner Cases

| Boundary ID | Test Condition | Expected Behavior |
|---|---|---|
| **B1. Slate Length Scale** | $N \in \{1, 2, 5, 20, 50, 100\}$ | Slates have exact shape $(n_u, N)$, quotas sum to $N$, no index out-of-bounds. |
| **B2. Extreme Inclinations**| $\text{Pop}_u = 0.0$ (pure tail) and $\text{Pop}_u = 1.0$ (pure head) | $N_{\text{pop}} = 0$ for tail-only user; $N_{\text{pop}} = N$ for head-only user. |
| **B3. Empty History / Cold Start** | User interaction row $M_{u, :} = \mathbf{0}$ ($|C_u| = 0$) | Fallback cleanly without ZeroDivisionError; return top-$N$ valid candidates. |
| **B4. Degenerate Head Pool**| Single-item head pool $|H| = 1$ | When $N_{\text{pop}} > 1$, head pool exhausts after 1 item and gracefully falls back to tail pool without duplication. |
| **B5. Numerical Score Extremes** | Uniform flat scores ($\sigma=0$), Infinite/NaN masks, Massive score gap ($\pm 10^6$) | Standardization handles $\sigma=0$ by zeroing; $- \infty$ items are strictly never selected; extreme gaps correctly trigger confidence elasticity. |
| **B6. Catalog Starvation** | Total available unseen items $< N$ | Return all available items padded safely without throwing unhandled exceptions. |
| **B7. Full Interaction User** | User has already interacted with all items in Head pool | Head quota gracefully rolls over to Tail pool without leaking training positives. |

---

### Tier 3: Cross-Feature Combinations

1. **Stochastic Quota + User Adaptive Alpha**:
   - Adaptive alpha shifts baseline $\text{Pop}_u \to \widetilde{\text{Pop}}_u$, which is subsequently passed to `stochastic_user_quota`.
   - Verified: Quota sum $N_{\text{pop}} + N_{\text{tail}} = N$ is strictly invariant, while expected head allocation reflects personalized alpha.
2. **Score Fusion + Confidence Elasticity**:
   - SBERT semantic similarity scores fused via `calibrated_score_fusion` at $\lambda=0.70$ are fed directly into `confidence_elastic_merge`.
   - Verified: Fused scores correctly drive confidence margin calculations and trigger item swaps.
3. **Dynamic Merge + Full-Rank Metric Evaluation**:
   - Scores processed through dual FedNCF backbones + LLM score fusion $\to$ merged into slates via `dynamic_probabilistic_quota_merge` $\to$ evaluated via `full_rank_eval(method='reclist')`.
   - Verified: Full metrics dictionary is populated with valid numeric values ($\text{Recall@20}$, $\text{NDCG@20}$, $\text{Rmse-PC}$, $\text{MRMC}$, $\text{GKPI}$, $\text{APLT}$, $\text{LTC}$).
4. **Hybrid Merge Strategy Verification**:
   - Tests mode `'hybrid'` integrating all three innovations simultaneously: Adaptive Alpha + Stochastic Rounding + Confidence Margin Interleaving.

---

### Tier 4: Real-World Workload Scenarios & Acceptance Criteria

1. **End-to-End ML-1M Scale Simulation Benchmark**:
   - Simulates $n_{\text{users}} = 6{,}040$, $n_{\text{items}} = 3{,}533$, Pareto volume threshold $\alpha = 0.20$ ($|H| \approx 74$ items), $\lambda = 0.70$ Sentence-BERT embeddings.
   - Evaluates at list length $N = 20$.
   - **Acceptance Assertions**:
     $$\text{Recall@20} > 0.0370 \quad (\text{Strict Requirement})$$
     $$\text{Rmse-PC} \le 0.0560 \quad (\text{Strict Requirement})$$
     $$\text{APLT} \ge 0.7000 \quad (\text{Long-Tail Exposure Guarantee})$$
2. **High-Concurrency Batch Scalability**:
   - Evaluates throughput of dynamic merge algorithms across 10,000 synthetic users to ensure sub-second inference runtime suitable for federated on-device deployment.

---

## 3. Mathematical Reference Formulas (Authoritative Oracles)

### 3.1 Expectation-Preserving Stochastic Quota
For user inclination $\text{Pop}_u \in [0, 1]$ and slate length $N$:
$$\text{frac}_u = N \cdot \text{Pop}_u - \lfloor N \cdot \text{Pop}_u \rfloor$$
$$N_{\text{pop}}(u) = \lfloor N \cdot \text{Pop}_u \rfloor + \mathbb{I}(U_u < \text{frac}_u), \quad U_u \sim \text{Uniform}(0, 1)$$
$$N_{\text{tail}}(u) = N - N_{\text{pop}}(u)$$
$$\mathbb{E}[N_{\text{pop}}(u)] = N \cdot \text{Pop}_u$$

### 3.2 User-Adaptive Alpha Thresholding
$$w_u = \frac{\log(1 + |C_u|)}{\max_{v} \log(1 + |C_v|)}$$
$$\text{Pop}_u^{\text{adaptive}} = \text{clip}\left( \text{Pop}_u \cdot (1 + \beta \cdot (w_u - 0.5)), \ 0.0, \ 1.0 \right)$$

### 3.3 Calibrated Score Fusion
For pool $P \in \{H, T\}$:
$$\widehat{S}_{\text{fed}}(u, i) = \frac{S_{\text{fed}}(u, i) - \mu_P(u)}{\sigma_P(u) + \epsilon}$$
$$\widetilde{S}_P(u, i) = (1 - \lambda) \cdot \widehat{S}_{\text{fed}}(u, i) + \lambda \cdot S_{\text{LLM}}(u, i) \quad \forall i \in P$$

### 3.4 Popularity Calibration RMSE (Rmse-PC)
$$\text{Rmse-PC} = \sqrt{\frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \left( \frac{|\text{RecList}(u) \cap H|}{N} - \text{Pop}_u \right)^2}$$

---

## 4. Test Execution Framework

- **Test Runner**: `pytest`
- **Execution Command**:
  ```powershell
  python -m pytest tests/test_h9_advanced_merge.py -v
  ```
- **Test File Location**: `c:/d_drive/projects/Project1/research/tests/test_h9_advanced_merge.py`
- **Output Artifact**: `TEST_READY.md` containing execution summary and verification checklist.
