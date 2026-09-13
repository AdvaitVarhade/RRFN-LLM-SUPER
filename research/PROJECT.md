# Project: FedSUPER-LLM Dynamic Blueprint Merge

## Architecture
FedSUPER-LLM is a federated recommender system that balances recommendation accuracy, user privacy, and popularity calibration on long-tail items. The system partitions catalog items into Head and Tail pools using Pareto volume thresholding ($\alpha = 0.20$), trains two decentralized matrix factorization / neural collaborative filtering models ($M_{\text{pop}}$ and $M_{\text{tail}}$) with client-private user embeddings, blends semantic similarity scores from Sentence-BERT user and item profiles ($\lambda = 0.70$), and merges candidate pools into personalized recommendation slates using user interaction blueprints.

The novel Dynamic Merge framework addresses the historical accuracy ceiling caused by static integer floor quantization ($N_{\text{pop}} = \lfloor N \cdot \text{Pop}_u \rfloor$). By introducing expectation-preserving stochastic rounding, user-adaptive alpha thresholds ($\alpha_u$), confidence-elastic boundary swaps, and Lagrangian dual knapsack optimization, the merge layer dynamically allocates slots based on prediction confidence while strictly preserving popularity calibration ($\text{Rmse-PC} \le 0.056$).

```
[ML-1M Dataset] ────────► [Pareto Partition] ──► Head Pool (H) & Tail Pool (T)
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
[Head Recommender M_pop]                     [Tail Recommender M_tail]
(Sparse FedAvg / BPR)                        (Sparse FedAvg / BPR)
        │                                               │
        ├───────────────────────┬───────────────────────┤
        ▼                       ▼                       ▼
 [SBERT Embeddings]      [Dynamic Merge Layer]    [User History Blueprint]
 (Semantic Fusion λ=0.7)  - Stochastic Quotas     (Local Inclination Pop_u)
                          - Adaptive Alpha
                          - Confidence Elastic
                                │
                                ▼
                   [Personalized Top-K Slate]
             (Recall@20 > 0.037, Rmse-PC <= 0.056)
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | `stochastic_user_quota` | Expectation-preserving randomized rounding for head/tail integer quotas | M1 | Survey (Explorer 1, 2, 3) |
| 2 | `user_adaptive_alpha_partition` | Personalized catalog thresholding based on user historical activity | M1 | Survey (Explorer 1, 2, 3) |
| 3 | `confidence_elastic_merge` | Score-gap based quota elasticity allowing high-confidence boundary item rescue | M1 | Survey (Explorer 1, 2, 3) |
| 4 | `dynamic_probabilistic_quota_merge` | Primary dynamic probabilistic blueprint merge algorithm supporting multiple quota modes | M1 | Survey (Explorer 1, 2, 3) |
| 5 | `calibrated_score_fusion` | Intra-pool standardized fusion of collaborative and SBERT semantic scores | M1 | Survey (Explorer 1, 2) |
| 6 | Unit Test Suite for Merge Functions | Unit tests verifying quota sums, expectation preservation, and calibration bounds | M1, E2E | Survey |
| 7 | H9 Standalone Experiment Harness (`run_h9.py`) | End-to-end experiment pipeline loading ML-1M, running FL / cached models, applying dynamic merge, evaluating metrics | M2 | Survey (Explorer 2) |
| 8 | Multi-Variant Strategy Comparison | Comparative benchmark across 7 variants (Static, Dynamic Probabilistic, Adaptive Alpha, Confidence Elastic, Hybrid) | M2 | Survey (Explorer 2) |
| 9 | JSON Metrics Serializer & Logger | Structured metric extraction (`metrics_h9.json`) and formatted terminal tables | M2 | Survey (Explorer 2) |
| 10 | Acceptance Criteria Verification Block | Programmatic assertions for Recall@20 > 0.037 and Rmse-PC <= 0.056 | M2, M3 | Survey (Explorer 2, 3) |
| 11 | Full E2E Benchmarking & Protocol Artifacts | Execution of H9, generating `protocol.md`, `analysis.md`, and `metrics_h9.json` | M3 | Survey |
| 12 | Opaque-Box E2E Testing Suite | Independent 4-Tier test harness validating all merge algorithms, edge cases, and CLI options | E2E-Track | Dual-Track Requirement |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Core Algorithmic Improvement (`src/super.py`) | Implement stochastic quotas, user-adaptive alpha, confidence elasticity, and dynamic blueprint merge functions in `src/super.py` | Survey complete | DONE |
| M2 | Standalone Experiment Harness (`run_h9.py`) | Create `experiments/H9-advanced-merge/code/run_h9.py` with multi-variant comparisons and JSON logging | M1 | DONE |
| M3 | E2E Execution, Verification & Analysis | Execute `run_h9.py`, verify Recall@20 > 0.037 and Rmse-PC <= 0.056, write experiment artifacts | M2 | DONE |
| E2E | Independent E2E Test Suite (`test_e2e_merge.py`) | Create 4-tier opaque-box test harness validating all merge algorithms, edge cases, and publish `TEST_READY.md` | Survey complete | DONE |

## Interface Contracts

### `src/super.py` ↔ `run_h9.py`
- `stochastic_user_quota(pop_u: np.ndarray, N: int = 20, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]`
  - Inputs: `pop_u` shape `(n_users,)` float in `[0, 1]`, `N` int.
  - Outputs: `(n_pop, n_tail)` shape `(n_users,)` int64, where `n_pop + n_tail == N` and `E[n_pop] == N * pop_u`.
- `user_adaptive_alpha_partition(pop_count: np.ndarray, train_matrix: np.ndarray, alpha_base: float = 0.20, beta: float = 0.15) -> np.ndarray`
  - Inputs: interaction counts `pop_count`, interaction matrix `train_matrix`.
  - Outputs: user-specific popularity inclinations `pop_u_adaptive`.
- `dynamic_probabilistic_quota_merge(scores_pop: np.ndarray, scores_tail: np.ndarray, train_matrix: np.ndarray, head_idx: np.ndarray, N: int = 20, mode: str = 'stochastic', seed: int = 42, **kwargs) -> np.ndarray`
  - Inputs: `scores_pop` (masked on tail), `scores_tail` (masked on head), `train_matrix`, `head_idx`, `N`.
  - Outputs: `reclist_matrix` shape `(n_users, N)` int64 containing recommended item indices.

### `src/metrics.py` ↔ `run_h9.py`
- `full_rank_eval(scores: Optional[np.ndarray], test: Dict[int, List[int]], train_matrix: np.ndarray, K: int = 20, head_idx: Optional[np.ndarray] = None, method: str = 'reclist', reclist_matrix: Optional[np.ndarray] = None) -> Dict[str, float]`
  - Outputs dictionary with keys: `recall@K`, `ndcg@K`, `aplt`, `ltc`, `rmse_pc`, `mrmc`, `gini_fairness`, `coverage`, `gkpi`.

## Code Layout
- `src/super.py`: Core blueprint merge and popularity calibration algorithms (M1 - Verified & Approved).
- `src/metrics.py`: Metric evaluation formulas (Read-only reference).
- `src/recdata.py`: Dataset loading and splitting (Read-only reference).
- `src/train.py`: Federated training functions (Read-only reference).
- `experiments/H9-advanced-merge/code/run_h9.py`: Main standalone experiment script (M2 - Verified & Approved).
- `experiments/H9-advanced-merge/results/metrics_h9.json`: Experiment output metrics (M3 - Verified & Approved).
- `experiments/H9-advanced-merge/protocol.md`: Experiment design and hypotheses (M3 - Verified & Approved).
- `experiments/H9-advanced-merge/analysis.md`: Analysis of results and findings (M3 - Verified & Approved).
- `tests/test_super_unit.py`: Unit test suite (M1 - Verified & Approved).
- `tests/test_h9_advanced_merge.py`: Independent 4-Tier E2E test harness (E2E Track - Verified & Approved).
