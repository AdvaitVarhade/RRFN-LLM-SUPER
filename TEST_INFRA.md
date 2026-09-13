# E2E Test Infra: Privacy-Preserving Federated SUPER Simulation & Dashboard

## Test Philosophy
- Opaque-box, requirement-driven. Derives strictly from `ORIGINAL_REQUEST.md`.
- Methodology: Category-Partition + Boundary Value Analysis (BVA) + Pairwise Combinatorial Testing + Real-World Workload Simulation.
- Headless verification: `test_app.py` and `tests/` execute 100% offline with zero external network dependencies.

## Feature Inventory & Test Coverage
| # | Feature | Source (Requirement) | Tier 1 (Units) | Tier 2 (Boundary) | Tier 3 (Cross) | Tier 4 (E2E) |
|---|---------|----------------------|:--------------:|:-----------------:|:--------------:|:------------:|
| 1 | Native Synthetic Mock Data Generator | ORIGINAL_REQUEST R1 | 5 | 5 | 2 | 2 |
| 2 | Privacy-Preserving Federated Core | ORIGINAL_REQUEST R1/R2 | 5 | 5 | 2 | 2 |
| 3 | SUPER Blueprint Merge & LLM Fusion | ORIGINAL_REQUEST R1/R3 | 5 | 5 | 2 | 2 |
| 4 | Evaluation Metrics ($Rmse\text{-}PC$, Gini) | ORIGINAL_REQUEST R3 | 5 | 5 | 2 | 2 |
| 5 | Network Flow Animation Visualizer (R2) | ORIGINAL_REQUEST R2 | 5 | 5 | 2 | 2 |
| 6 | Popularity Calibration Charts (R3) | ORIGINAL_REQUEST R3 | 5 | 5 | 2 | 2 |
| 7 | Interactive Control Dashboard (R1) | ORIGINAL_REQUEST R1 | 5 | 5 | 2 | 2 |
| 8 | Headless Port Binding & Verification | ORIGINAL_REQUEST Acceptance | 5 | 5 | 2 | 2 |

## Test Architecture
- Test runner: `pytest tests/ -v` and `python test_app.py`
- Directory layout:
  ```
  c:/d_drive/projects/Project1/fedsuper_simulation/
  ├── test_app.py
  └── tests/
      ├── test_tier1_feature_units.py
      ├── test_tier2_boundary_corner.py
      ├── test_tier3_cross_feature.py
      └── test_tier4_real_world_e2e.py
  ```

## Acceptance Assertions & Pass/Fail Thresholds
1. **$Rmse\text{-}PC$ Guarantee**: Calibrated recommendations satisfy $Rmse\text{-}PC \le 0.060$ (uncalibrated baseline $\ge 0.350$).
2. **Gini Index Reduction**: Calibrated exposure Gini $\le 0.400$ (uncalibrated baseline $\ge 0.700$).
3. **Long-Tail Exposure Ratio**: $LTER \ge 0.250$ (uncalibrated baseline $\le 0.050$).
4. **Network Flow Visual Channels**: Graph figure contains distinct trace colors for Local Private Data (`#FF7043`), DP Gradients (`#00E5FF`), and LLM Blueprints (`#D500F9`).
5. **Headless HTTP Startup**: Application loads and responds `HTTP 200` to health check without startup exceptions.

## Coverage Thresholds
- Tier 1 (Feature & Component Units): $\ge 40$ test cases
- Tier 2 (Boundary & Corner Cases): $\ge 40$ test cases
- Tier 3 (Cross-Feature & State Machine): $\ge 15$ test cases
- Tier 4 (Real-World & E2E System): $\ge 10$ test cases
- **Total Minimum Test Cases: $\ge 105$ test cases**
