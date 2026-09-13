# Project: Privacy-Preserving Federated SUPER Dashboard and Simulation

## Architecture
The system is an interactive, standalone web dashboard and simulation platform for the Privacy-Preserving Federated SUPER (Smart User-centric Popularity Exposure Reduction) recommendation pipeline.
- **Frontend / Dashboard**: Streamlit web interface with reactive controls (Step, Play/Pause, Reset, Hyperparameter Sliders) and Plotly interactive visualizations.
- **Visualization Engine**:
  - **Network Flow Graph (R2)**: Radial coordinate topology with Central Server at center, LLM Knowledge Base, and client devices. Features strict visual differentiation:
    1. *Local Private Data*: Orange glowing boundary halo around client nodes (Zero Egress).
    2. *DP-Clipped Gradients*: Cyan neon directed transmission arrows between active clients and server.
    3. *LLM Semantic Blueprints*: Magenta energy stream from LLM Knowledge Base to server.
  - **Popularity Calibration Charts (R3)**: Head vs Torso vs Tail grouped bar chart, log-scale rank-exposure curves, $Rmse\text{-}PC$ error distributions, and multi-objective Pareto frontiers (Accuracy vs Fairness vs Privacy).
- **Federated Simulation Engine**:
  - Pure NumPy/SciPy lightweight standalone engine running sparse FedAvg over global item embeddings $\mathbf{Q} \in \mathbb{R}^{M \times d}$ and biases $\mathbf{b} \in \mathbb{R}^M$.
  - Local client privacy enclaves holding private user embeddings $\mathbf{p}_u$, interaction histories $D_u$, and LLM semantic profiles $\mathbf{s}_u$.
  - Differential privacy engine with $L_2$ gradient clipping ($C$) and Gaussian noise perturbation ($\sigma$).
  - SUPER blueprint merge engine with intra-pool z-score standardization of collaborative filtering and LLM semantic cosine similarities.
- **Synthetic Data Generator**:
  - Fully offline, standalone generation of catalog items, power-law popularity distribution, client cohorts, Dirichlet genre preferences, and LLM semantic embedding vectors.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Native Synthetic Mock Data Generator | Generates client cohort, items, power-law interactions, genre centroids, LLM semantic profiles/embeddings, and Head/Torso/Tail partitions entirely offline. | M1 | Survey 1, ORIGINAL_REQUEST R1 |
| 2 | Privacy-Preserving Federated Core Engine | Sparse FedAvg simulation with client-side user embeddings, server-side item parameters, local BPR loss, $L_2$ gradient clipping ($C$), and Gaussian DP noise ($\sigma$). | M1 | Survey 1, ORIGINAL_REQUEST R1/R2 |
| 3 | SUPER Blueprint Calibration & LLM Fusion | Intra-pool z-score standardization fusing CF and LLM semantic similarities, user historical inclination computation $P_u$, and calibrated quota allocation. | M1 | Survey 1, Survey 3, ORIGINAL_REQUEST R1/R3 |
| 4 | Comprehensive Evaluation Metrics Engine | Computes $Rmse\text{-}PC$, Gini exposure index $G$, Long-tail Exposure Ratio ($LTER$), Catalog Coverage ($Cov$), Recall@K, NDCG@K, Novelty, and ILD. | M1 | Survey 1, Survey 3, ORIGINAL_REQUEST R3 |
| 5 | Network Flow Animation & Graph Visualizer | 2D interactive Plotly graph visually differentiating strictly local private data (Orange), DP-clipped gradients (Cyan), and LLM semantic blueprints (Magenta). | M2 | Survey 2, ORIGINAL_REQUEST R2 |
| 6 | Popularity Calibration & Analytical Charts | Plotly charts for Head/Torso/Tail exposure comparison, log-scale rank-exposure distribution, $Rmse\text{-}PC$ distribution, and Pareto front. | M2 | Survey 2, Survey 3, ORIGINAL_REQUEST R3 |
| 7 | Interactive Streamlit Control Dashboard | Web UI with Play/Pause/Step/Reset controls, round slider, hyperparameter tuning ($\alpha, \lambda, \sigma, S$), KPI cards, and client inspector. | M3 | Survey 2, ORIGINAL_REQUEST R1 |
| 8 | Headless Verification Harness & Test Suite | Standalone `test_app.py` and 4-tier pytest suite verifying mock data generation, port binding, UI rendering functions, and $Rmse\text{-}PC \le 0.060$. | E2E Track & Final | Survey 3, ORIGINAL_REQUEST Acceptance |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| E2E | E2E Testing Track | Comprehensive 4-tier test suite (`fedsuper_simulation/tests/`) and standalone verification script (`test_app.py`) publishing `TEST_READY.md`. | none | DONE (f2e64d53-b020-4d23-8b0f-99c7cbfeb814) |
| M1 | FedSUPER Core Simulation & Mock Data | `config.py`, `mock_data.py`, `federated_core.py`, `super_engine.py`, `evaluator.py`. | none | DONE (7a5f9cfa-8e53-45c6-a43a-c7a47ced14a0) |
| M2 | Visualizations & Graph Flow Engines | `graph_visualizer.py`, `chart_generator.py`. | M1 (Interface contracts) | DONE (f3451cda-0c16-4468-862d-6d3fb814db44) |
| M3 | Interactive Web Dashboard Application | `app.py` (Streamlit web application, state machine, controls, layout tabs). | M1, M2 | DONE (f3451cda-0c16-4468-862d-6d3fb814db44) |
| M4 | Final Milestone: 100% E2E Pass & Adversarial Hardening | Verify 100% pass on Tiers 1-4, execute Tier 5 adversarial stress testing, independent audit. | E2E, M1, M2, M3 | DONE (Gate PASS) |

## Code Layout
```
c:/d_drive/projects/Project1/fedsuper_simulation/
├── app.py                      # Main interactive Streamlit application entrypoint (M3)
├── requirements.txt            # Python dependencies (streamlit, plotly, networkx, numpy, pandas, scipy, scikit-learn)
├── test_app.py                 # Standalone E2E verification script (E2E Track)
├── src/
│   ├── __init__.py
│   ├── config.py               # SimulationConfig dataclass (M1)
│   ├── mock_data.py            # Native synthetic data generator (M1)
│   ├── federated_core.py       # Standalone FL engine with DP gradient clipping/noise (M1)
│   ├── super_engine.py         # Popularity blueprint merge & LLM semantic fusion (M1)
│   ├── evaluator.py            # Beyond-accuracy metrics (Rmse-PC, Gini, LTER, Coverage, etc.) (M1)
│   ├── graph_visualizer.py     # Plotly & NetworkX federated topology visualizer (M2)
│   └── chart_generator.py      # Plotly analytical charts (M2)
└── tests/
    ├── __init__.py
    ├── test_tier1_feature_units.py
    ├── test_tier2_boundary_corner.py
    ├── test_tier3_cross_feature.py
    └── test_tier4_real_world_e2e.py
```

## Interface Contracts

### 1. `config.py`
```python
@dataclass
class SimulationConfig:
    num_clients: int = 20
    num_items: int = 100
    embedding_dim: int = 32
    llm_dim: int = 32
    pareto_alpha: float = 0.20
    calibration_alpha: float = 0.40
    llm_lambda: float = 0.70
    dp_enabled: bool = True
    dp_epsilon: float = 4.0
    dp_delta: float = 1e-5
    dp_l2_clip_norm: float = 1.0
    top_k: int = 10
    clients_per_round: int = 5
    learning_rate: float = 0.05
    max_rounds: int = 20
```

### 2. `mock_data.py` ↔ `federated_core.py` / `super_engine.py`
```python
@dataclass
class SyntheticDataset:
    num_users: int
    num_items: int
    user_metadata: pd.DataFrame       # user_id, preferred_genres, pop_u_true
    item_metadata: pd.DataFrame       # item_id, title, category, popularity_count, is_head, is_torso, is_tail
    train_matrix: np.ndarray          # (num_users, num_items) binary interaction matrix
    test_dict: Dict[int, List[int]]   # user_id -> held-out items
    user_llm_profiles: np.ndarray     # (num_users, llm_dim) normalized semantic vectors
    item_llm_embeddings: np.ndarray   # (num_items, llm_dim) normalized semantic vectors
    head_idx: np.ndarray              # Indices of Head items (top 20%)
    torso_idx: np.ndarray             # Indices of Torso items (next 30%)
    tail_idx: np.ndarray              # Indices of Tail items (bottom 50%)

def generate_mock_dataset(config: SimulationConfig, seed: int = 42) -> SyntheticDataset:
    """Generates synthetic client-item interaction matrix, genre clusters, and semantic vectors."""
```

### 3. `federated_core.py` & `super_engine.py` ↔ `evaluator.py` & `app.py`
```python
class FederatedSimulation:
    def __init__(self, dataset: SyntheticDataset, config: SimulationConfig):
        ...
    def step_round(self) -> Dict[str, Any]:
        """Advances one federated round: samples clients, performs local DP training, aggregates deltas, executes blueprint merge."""
    def get_recommendations(self, calibrated: bool = True) -> np.ndarray:
        """Returns (num_users, top_k) matrix of recommended item IDs."""
```

### 4. `evaluator.py` ↔ `chart_generator.py` & `app.py`
```python
def calculate_rmse_pc(user_blueprints: np.ndarray, rec_distributions: np.ndarray) -> float:
    """Calculates macro-averaged Root Mean Squared Error of Popularity Calibration."""

def calculate_gini_index(exposure_counts: np.ndarray) -> float:
    """Calculates Gini coefficient of item recommendation exposures."""

def calculate_long_tail_exposure_ratio(exposure_counts: np.ndarray, tail_idx: np.ndarray) -> float:
    """Calculates fraction of total recommendations allocated to tail items."""

def evaluate_all_metrics(dataset: SyntheticDataset, recs_uncalib: np.ndarray, recs_calib: np.ndarray, top_k: int) -> Dict[str, Any]:
    """Returns comprehensive dictionary of metrics comparing uncalibrated vs SUPER calibrated."""
```

### 5. `graph_visualizer.py` ↔ `app.py`
```python
def render_network_flow_figure(
    dataset: SyntheticDataset,
    active_client_ids: List[int],
    current_round: int,
    dp_clip_norm: float,
    llm_lambda: float,
    theme: str = "plotly_dark"
) -> go.Figure:
    """Returns interactive Plotly Figure with 3 visually differentiated streams: Local Data (Orange), DP Gradients (Cyan), LLM Blueprints (Magenta)."""
```

### 6. `chart_generator.py` ↔ `app.py`
```python
def render_exposure_comparison_chart(metrics: Dict[str, Any], theme: str = "plotly_dark") -> go.Figure:
    """Returns grouped bar chart of Head vs Torso vs Tail exposure for Blueprint vs Uncalibrated vs Calibrated."""

def render_rank_exposure_curve(exposure_uncalib: np.ndarray, exposure_calib: np.ndarray, theme: str = "plotly_dark") -> go.Figure:
    """Returns log-scale rank vs exposure area curve."""

def render_pareto_front_chart(pareto_records: List[Dict[str, float]], current_alpha: float, theme: str = "plotly_dark") -> go.Figure:
    """Returns Accuracy (RMSE/Recall) vs Fairness (Rmse-PC/Gini) Pareto scatter."""

def render_convergence_chart(history_records: List[Dict[str, float]], theme: str = "plotly_dark") -> go.Figure:
    """Returns training loss and RMSE curves over rounds."""
```
