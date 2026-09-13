# Experiment Protocol: H9 — Advanced Dynamic Blueprint Merge for FedSUPER-LLM

**Experiment ID**: `H9-advanced-merge`  
**Date**: 2026-08-18  
**Status**: Pre-registered & Executed  

---

## 1. Context & Motivation

FedSUPER (Yavru et al., IEEE Access 2026) introduced a decentralized dual-model architecture ($M_{\text{pop}}$ and $M_{\text{tail}}$) combined with a blueprint merge layer to balance accuracy, privacy, and popularity calibration on long-tail items. However, the static blueprint merge relies on integer floor quantization ($N_{\text{pop}} = \lfloor N \cdot \text{Pop}_u \rfloor$). 

This introduces three critical bottlenecks:
1. **Integer Floor Distortion**: Users with popularity inclination $\text{Pop}_u \in (0, 1/N)$ have their head quota truncated to zero ($N_{\text{pop}} = 0$), completely excluding relevant head candidates.
2. **Rigid Catalog Partitioning**: A fixed global cumulative volume threshold ($\alpha = 0.20$) ignores the breadth of individual user histories, penalizing active vs. sparse users uniformly.
3. **Lack of Confidence Elasticity**: When the scheduled pool candidate scores degrade into noise while the alternative pool holds high-confidence recommendations, static merge cannot swap boundary slots.

Prior continuous score-penalty approaches (e.g., Logit-Adjusted calibration in H8) failed catastrophically (Recall dropped to $0.024$ and Rmse-PC exploded to $0.150$) because modifying raw scores corrupts sensitive intra-pool ranking signals. 

H9 investigates whether **dynamic, probabilistic, and confidence-elastic blueprint merge algorithms** operating strictly at the inter-pool selection layer can surpass the optimal accuracy ceiling ($\text{Recall@20} > 0.037$) while strictly upholding popularity calibration guarantees ($\text{Rmse-PC} \le 0.056$).

---

## 2. Hypotheses (Locked)

- **Hypothesis 1 (Expectation-Preserving Quota Optimization)**:  
  Replacing integer floor quantization with stochastic randomized rounding will increase Recall by eliminating truncation bias on boundary users without altering the expected popularity calibration ($\mathbb{E}[\text{Rmse-PC}] \le 0.056$).
- **Hypothesis 2 (Personalized Breadth Adaptation)**:  
  Modulating Pareto catalog boundaries based on user activity levels ($\alpha_u = \alpha_{\text{base}}(1 + \beta(w_u - 0.5))$) will better align head/tail quotas with user catalog exposure, improving both accuracy (Recall@20) and item catalog coverage.
- **Hypothesis 3 (Confidence-Elastic Boundary Interleaving)**:  
  Allowing bounded 1-slot boundary swaps when inter-pool score margins exceed a threshold ($\Delta S > \theta$) will rescue high-confidence recommendations from the alternate pool while maintaining individual calibration error within the theoretical $1/N$ bound.
- **Hypothesis 4 (Synergistic LLM + Dynamic Merge)**:  
  Combining Sentence-BERT semantic similarity score fusion ($\lambda = 0.70$) with hybrid dynamic merge will achieve a compound accuracy lift over the FedSUPER baseline while strictly satisfying all fairness and calibration guarantees.

---

## 3. Algorithmic Formulations

### 3.1 Formulation 1: Expectation-Preserving Stochastic Quota (`stochastic_user_quota`)
For target list length $N$ and user inclination $\text{Pop}_u \in [0, 1]$:
$$\mu_{\text{pop}} = N \cdot \text{Pop}_u$$
$$k_{\text{floor}} = \lfloor \mu_{\text{pop}} \rfloor, \quad p_{\text{rem}} = \mu_{\text{pop}} - k_{\text{floor}}$$
$$N_{\text{pop}} = k_{\text{floor}} + \mathbb{I}(X_u < p_{\text{rem}}), \quad X_u \sim \text{Uniform}(0, 1)$$
$$N_{\text{tail}} = N - N_{\text{pop}}$$
- **Mathematical Guarantees**:
  - Exact sum invariance: $N_{\text{pop}} + N_{\text{tail}} = N, \quad \forall u$.
  - Unbiased expectation: $\mathbb{E}[N_{\text{pop}}] = N \cdot \text{Pop}_u$.
  - Strict individual deviation bound: $\left|\frac{N_{\text{pop}}}{N} - \text{Pop}_u\right| \le \frac{1}{N}$.

### 3.2 Formulation 2: User-Adaptive Alpha Catalog Partitioning (`user_adaptive_alpha_partition`)
Compute user historical activity weight:
$$w_u = \frac{\log(1 + |\mathcal{C}_u|)}{\max_v \log(1 + |\mathcal{C}_v|)}$$
$$\alpha_u = \text{clip}\left(\alpha_{\text{base}} \cdot \left(1 + \beta \cdot (w_u - 0.5)\right), 0.05, 0.50\right)$$
Derive user-specific popularity inclinations $\text{Pop}_u^{\text{adapt}}$ from personalized head thresholds $\mathcal{H}_u$, scaling catalog sensitivity dynamically.

### 3.3 Formulation 3: Confidence-Elastic Boundary Merge (`confidence_elastic_merge`)
For boundary candidates $c_{\text{head}} \in C_{\text{head}}[N_{\text{pop}}]$ and $c_{\text{tail}} \in C_{\text{tail}}[N_{\text{tail}}-1]$:
$$\text{If } S_{\text{head}}[u, c_{\text{head}}] - S_{\text{tail}}[u, c_{\text{tail}}] > \theta \implies N_{\text{pop}} \leftarrow N_{\text{pop}} + 1, \; N_{\text{tail}} \leftarrow N_{\text{tail}} - 1$$
$$\text{If } S_{\text{tail}}[u, c_{\text{tail}}] - S_{\text{head}}[u, c_{\text{head}}] > \theta \implies N_{\text{tail}} \leftarrow N_{\text{tail}} + 1, \; N_{\text{pop}} \leftarrow N_{\text{pop}} - 1$$
Bounded elasticity preserves calibration within $|N_{\text{pop}}/N - \text{Pop}_u| \le 1/N + 1/N$.

### 3.4 Formulation 4: Intra-Pool Calibrated Score Fusion (`calibrated_score_fusion`)
For collaborative filtering scores $S_{\text{fed}}$ and SBERT profile similarity $S_{\text{LLM}} = U_{\text{emb}} I_{\text{emb}}^T$:
$$S_{\text{norm}}[u, i] = \frac{S_{\text{fed}}[u, i] - \mu_u^{\text{pool}}}{\sigma_u^{\text{pool}} + \epsilon}, \quad \forall i \in \text{Pool}$$
$$S_{\text{fused}}[u, i] = (1 - \lambda) S_{\text{norm}}[u, i] + \lambda S_{\text{LLM}}[u, i], \quad \lambda = 0.70$$

---

## 4. Experimental Setup

- **Dataset**: MovieLens-1M ($6,040$ users, $3,533$ items, $575,281$ interactions with rating $\ge 4$).
- **Partitioning**: Leave-one-out chronological split; Pareto cumulative volume threshold $\alpha = 0.20$ ($74$ head items, $3,459$ tail items).
- **Federated Architecture**:
  - Backbone: Matrix Factorization ($d = 64$).
  - Dual models: $M_{\text{pop}}$ trained on $\mathcal{D}_{\text{pop}}$, $M_{\text{tail}}$ trained on $\mathcal{D}_{\text{tail}}$.
  - Algorithm: Sparse FedAvg / BPR loss ($30$ rounds, $256$ clients/round, $2$ local epochs, $\text{lr} = 0.3$).
- **LLM Embeddings**: Sentence-BERT `all-MiniLM-L6-v2` ($384$ dimensions, normalized).
- **Evaluation Cutoffs**: $K = 20$ (primary benchmark) and $K = 10$ (backwards comparison).
- **Random Seed**: $42$ (fixed for deterministic reproducibility).

---

## 5. Comparative Variants

1. `FedSUPER-Static`: Baseline static integer floor blueprint merge (Yavru et al., 2026).
2. `FedSUPER-Dynamic-Stochastic`: Dynamic expectation-preserving stochastic quota merge.
3. `FedSUPER-Dynamic-AdaptiveAlpha`: User-adaptive alpha volume merge ($\beta = 0.15$).
4. `FedSUPER-Dynamic-ConfidenceElastic`: Confidence score-gap elastic merge ($\theta = 0.10$).
5. `LLM-FedSUPER-Static`: LLM semantic fusion ($\lambda = 0.70$) + static merge.
6. `LLM-FedSUPER-Dynamic-Stochastic`: LLM semantic fusion ($\lambda = 0.70$) + stochastic quota merge.
7. `LLM-FedSUPER-Hybrid`: LLM semantic fusion ($\lambda = 0.70$) + hybrid adaptive stochastic elastic merge.

---

## 6. Acceptance Criteria

| Criterion | Target | Metric Function |
|---|---|---|
| **C1: Script Execution** | Flawless completion | `run_h9.py` exit code 0 |
| **C2: Accuracy Target** | $\text{Recall@20} > 0.0370$ | `full_rank_eval['recall@K']` |
| **C3: Calibration Bound** | $\text{Rmse-PC} \le 0.0560$ | `full_rank_eval['rmse_pc']` |
| **C4: Long-Tail Floor** | $\text{APLT} \ge 0.7000$ | `full_rank_eval['aplt']` |
