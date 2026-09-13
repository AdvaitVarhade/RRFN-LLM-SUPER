# Original User Request

## 2026-08-19T06:01:47Z

# Teamwork Project Prompt

> Requested team: Full multi-agent team

Build a comprehensive interactive web dashboard and visual simulation for the Privacy-Preserving Federated SUPER pipeline. The application must feature an interactive web interface (e.g., Streamlit or Dash) that includes animated network graphs showing data flows (local user profiles, global item gradients, LLM semantic vector integration) between federated nodes and the central server, alongside charts demonstrating popularity calibration.

Working directory: c:/d_drive/projects/Project1/fedsuper_simulation
Integrity mode: benchmark

## Requirements

### R1. Interactive Control Dashboard
Develop a web-based dashboard that allows users to step through or play the federated training and blueprint merge simulation. It should generate mock data (clients, items, interactions, and semantic embeddings) natively so it can run entirely standalone.

### R2. Network Flow Animation
Implement a visual representation (e.g., using Plotly, NetworkX, or similar interactive charting libraries) of the federated architecture. It must visually differentiate between data that stays local (user interaction history) and data that is aggregated globally (item embeddings).

### R3. Popularity Calibration Visualization
Include analytical charts that compare the recommendation distribution of a standard, uncalibrated algorithm versus the SUPER blueprint-calibrated algorithm, highlighting the correction of popularity bias.

## Acceptance Criteria

### Execution & Stability
- [ ] The dashboard application starts up successfully and binds to a local port without throwing startup exceptions.
- [ ] A verification script (`test_app.py`) successfully loads the core application logic or sends a mock HTTP request verifying that the application renders without crashing.

### Core Functionality
- [ ] The application generates synthetic mock data upon startup without requiring external CSV/dataset downloads.
- [ ] The application contains explicit UI components or rendering functions for the visual network graph representing the client-server architecture.
- [ ] The application contains rendering functions for a comparative chart plotting popularity exposure (e.g., Head vs. Tail recommendation ratios).
