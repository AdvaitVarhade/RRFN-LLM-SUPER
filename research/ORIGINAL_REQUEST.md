# Original User Request

## Initial Request — 2026-08-18T17:31:23+05:30

You are the Project Orchestrator for this research & engineering task.

## Working Environment
- Project Root / Working Directory: c:/d_drive/projects/Project1/research
- Your Working Directory: c:/d_drive/projects/Project1/research/.agents/teamwork_preview_orchestrator_1
- Original Request File: c:/d_drive/projects/Project1/research/ORIGINAL_REQUEST.md
- Integrity Mode: demo

## Task Mission & Objective
Research, design, and implement a novel algorithmic improvement to the SUPER blueprint merge algorithm (e.g., dynamic probabilistic quotas, user-specific alpha thresholds) for the FedSUPER-LLM recommendation pipeline. The objective is to surpass the current optimal performance (Recall@20 of 0.037) while maintaining privacy and popularity calibration guarantees (Rmse-PC <= 0.056).

## Specific Requirements
### R1. Algorithmic Improvement
Design and implement a new variant of the SUPER blueprint merge algorithm in `src/super.py`. The focus should be on dynamic, probabilistic, or user-specific quota merges rather than the static integer quotas currently used.

### R2. Experiment Harness
Create a new standalone experiment script (`experiments/H9-advanced-merge/code/run_h9.py`). It must load the ML-1M dataset, utilize the existing federated training functions or cached scores, apply your new merge algorithm, and output metrics using `metrics.py`.

## Verification Resources & Acceptance Criteria
- Existing references: `experiments/H3-llm-profiling/code/run_h3.py` and `experiments/H1-fedsuper/code/run_h1_v2.py`.
- Acceptance Criteria:
  1. `run_h9.py` executes successfully from start to finish without errors.
  2. The printed evaluation metrics show a Recall@20 strictly greater than 0.037.
  3. The printed evaluation metrics show an Rmse-PC <= 0.056, proving popularity calibration is maintained.

Please organize your subagents/workers to research, implement, and rigorously verify this task. When completed, write your progress, handoff, and results, and report completion back to the Sentinel.
