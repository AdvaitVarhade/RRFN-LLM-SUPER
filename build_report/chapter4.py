"""
Chapter 4: System Implementation and Architecture Details
Comprehensive, publication-grade text for BCSE497J Project Report.
Includes candidate-specific software implementation emphasis callout blocks.
"""

from report_builder import (
    add_chapter_heading, add_section_heading, add_subsection_heading,
    add_paragraph, add_bullet, add_numbered, add_figure, add_table_custom,
    add_code_block, add_callout_box
)
from student_configs import STUDENT_PROFILES

def build_chapter_4(doc, student_config=None):
    if student_config is None:
        student_config = STUDENT_PROFILES[0]

    add_chapter_heading(doc, "CHAPTER 4", "SYSTEM IMPLEMENTATION AND ARCHITECTURE DETAILS")

    # 4.1 Software Architecture
    add_section_heading(doc, "4.1 Software Architecture and Design Principles")
    add_paragraph(
        doc,
        "The RRFN-LLM-SUPER codebase (robust_super) is engineered adhering strictly to enterprise-grade software architecture principles: "
        "Separation of Concerns, High Cohesion, Loose Coupling, Deterministic Reproducibility, and Hardware-Aware Vectorization. "
        "Implemented in Python 3.11 with PyTorch 2.x, NumPy, SciPy, and Streamlit, the system is partitioned into ten specialized packages:"
    )
    add_bullet(doc, "data: Multi-domain data ingestion, k-core graph filtering, temporal splitting, and adversarial attack injection engines.", bold_prefix="1. Data Pipeline: ")
    add_bullet(doc, "models: Deep collaborative filtering (NeuMF), Variational Autoencoders (VaeCF), and graph neural networks (LightGCN, SimGCL, SGL).", bold_prefix="2. Neural Backbones: ")
    add_bullet(doc, "rrfn: Consensus anchor selection, empirical rating transition matrix estimation, and Tikhonov regularized inversion.", bold_prefix="3. Noise Inversion: ")
    add_bullet(doc, "llm_auditor: Prompt engineering, LLM API interaction, structured JSON parsing, and SQLite WAL caching.", bold_prefix="4. LLM Semantic Auditing: ")
    add_bullet(doc, "bombing_detector: Temporal velocity acceleration, rating polarity skew, and 2D BLAS matrix burst detection.", bold_prefix="5. Burst Detector: ")
    add_bullet(doc, "fusion: Tri-view signal synthesis combining statistical, semantic, and burst views into sample reliability weights.", bold_prefix="6. Multi-View Fusion: ")
    add_bullet(doc, "catalog: Pareto volume partitioning dividing items into Head (M_pop) and Tail (M_tail) subsets.", bold_prefix="7. Catalog Partitioning: ")
    add_bullet(doc, "super_engine: Purified historical blueprints, personalized tail inclinations, and trajectory interleaving.", bold_prefix="8. Calibrated SUPER Engine: ")
    add_bullet(doc, "trainer: Dual partition training orchestrators, sample-reliability weighted InfoNCE losses, and optimizers.", bold_prefix="9. Training Engine: ")
    add_bullet(doc, "evaluation: Information retrieval metrics, popularity calibration metrics, attack detection metrics, and Monte Carlo economic simulation.", bold_prefix="10. Evaluation Engine: ")

    # 4.2 Module Mapping Table
    add_section_heading(doc, "4.2 Comprehensive Module Mapping and Source Code Index")
    add_paragraph(
        doc,
        "To provide full architectural transparency, Table 4.1 documents the complete source tree of the robust_super framework, detailing the exact "
        "class names, core functions, and algorithmic responsibilities of each module."
    )

    t41_headers = ["Module Path", "Primary Classes / Components", "Core Algorithmic Responsibility"]
    t41_data = [
        ["data/loader.py", "MovieLensLoader, InteractionSplit", "MovieLens-1M sub-graph ingestion, k-core pruning, and zero-leakage temporal splitting."],
        ["data/amazon_loader.py", "AmazonReviewLoader", "5-core Amazon Electronics loader extracting ratings, timestamps, and natural language reviews."],
        ["data/yelp_loader.py", "YelpLoader", "5-core Yelp Academic dataset ingestion with multi-modal metadata and review text."],
        ["data/attack_simulator.py", "AttackSimulator, AttackType", "Parameterized injection of Bandwagon Shilling attacks, Random noise, and Review-Bombing."],
        ["models/neumf.py", "NeuMF, GMF, MLP", "Neural Matrix Factorization combining Generalized Matrix Factorization and non-linear MLP."],
        ["models/simgcl.py", "SimGCL, SimGCLLayer", "Simple Graph Contrastive Learning generating views via latent uniform noise perturbations."],
        ["models/sgl.py", "SGL, EdgeDropoutLayer", "Self-supervised Graph Learning generating views via structural graph edge dropout."],
        ["models/lightgcn.py", "LightGCN, GraphConv", "Linearized graph convolutional message-passing backbone without non-linearities."],
        ["models/vaecf.py", "VaeCF, MultVAE", "Variational Autoencoder for collaborative filtering with float32 logvar clamping."],
        ["rrfn/anchor_selector.py", "AnchorSelector", "Identifies consensus rating anchors across users satisfying separability Assumption 3.1."],
        ["rrfn/transition_matrix.py", "TransitionMatrixEstimator", "Estimates KxK stochastic rating transition matrix T and computes Tikhonov pseudo-inverse."],
        ["rrfn/risk_loss.py", "RiskConsistentLoss", "Computes forward-corrected surrogate cross-entropy loss regularized against anchor prior."],
        ["llm_auditor/auditor.py", "LLMReviewAuditor", "Asynchronous LLM prompt evaluation scoring review-rating alignment and bot phrasing."],
        ["llm_auditor/cache.py", "SQLiteLLMCache", "ACID-compliant SQLite database with Write-Ahead Logging (WAL) for O(1) audit retrieval."],
        ["bombing_detector/bomb_scorer.py", "ReviewBombingDetector", "Hardware-accelerated 2D BLAS matrix burst detection combining Z_temp and S_pol (<5ms)."],
        ["fusion/reliability_fusion.py", "MultiViewReliabilityFilter", "MIRF tri-view ensemble layer synthesizing R_RRFN, R_LLM, and R_bomb into weights w_ui."],
        ["catalog/pareto_partition.py", "ParetoCatalogPartitioner", "Partitions catalog at rho=0.20 into Head (I_head) and Tail (I_tail) item subsets."],
        ["super_engine/blueprint.py", "PurifiedBlueprintBuilder", "Purifies historical user blueprints by filtering edges below trust threshold theta_trust."],
        ["super_engine/inclination.py", "TailInclinationCalculator", "Computes personalized tail inclination alpha_u using normalized reliability weights."],
        ["super_engine/merger.py", "RankCalibratedMerger", "Assembles top-K recommendations via rank-calibrated trajectory interleaving."],
        ["trainer/dual_trainer.py", "DualPartitionTrainer", "Concurrent training orchestrator for Head and Tail models under joint risk and contrastive losses."],
        ["trainer/contrastive_loss.py", "SampleWeightedInfoNCE", "InfoNCE contrastive loss modulated by sample-level reliability weights w_i."],
        ["evaluation/metrics.py", "RecommendationMetrics", "Evaluates Recall@K, nDCG@K, RMSE-PC, MRMC, LTC@K, and General KPI (GKPI)."],
        ["evaluation/robustness_metrics.py", "RobustnessMetrics", "Evaluates Genuine Long-Tail Coverage (g-LTC), Spam Target Exposure (STER), ROC, and AP."],
        ["evaluation/ab_simulator.py", "OfflineABSimulator", "1,000-sample Monte Carlo bootstrap economic simulator projecting CTR, CVR, and GMV."],
        ["dashboard.py", "StreamlitDashboard", "Interactive 4-panel web dashboard providing real-time telemetry, heatmaps, and user inspection."]
    ]
    t41_widths = [4.2, 4.4, 5.9]
    add_table_custom(
        doc, t41_headers, t41_data,
        caption_text="Comprehensive Software Architecture and Module Mapping Across the robust_super Framework.",
        table_num_str="4.1",
        col_widths=t41_widths
    )

    # 4.3 Data Pipeline
    add_section_heading(doc, "4.3 Multi-Domain Data Ingestion and Robust Loading Pipelines")
    add_paragraph(
        doc,
        "Data loading in robust_super is designed to prevent data leakage and handle multi-modal edge cases gracefully. The pipeline supports three distinct domains:"
    )
    add_paragraph(
        doc,
        "1. MovieLens-1M Loader (loader.py): Ingests 1,000,209 ratings from GroupLens. To construct a computationally dense evaluation graph, "
        "it executes iterative k-core pruning (k_user >= 10, k_item >= 5) and samples the top 300 active users, yielding |U| = 300 users, |I| = 3,405 items, "
        "and 237,421 clean training interactions. Data splitting strictly precedes attack injection using leave-one-out temporal ordering: each user's "
        "chronologically final interaction is assigned to the test set, penultimate to validation, and remaining to training. Adversarial attacks are injected "
        "exclusively into the training graph, guaranteeing zero test set contamination.",
        bold_prefix="MovieLens Pipeline: "
    )
    add_paragraph(
        doc,
        "2. Amazon Reviews Loader (amazon_loader.py): Ingests the Amazon Electronics dataset (Ni et al., 2019), extracting numeric ratings (1–5), "
        "Unix timestamps, and full review text. A 5-core filter yields |U| = 60 users, |I| = 81 items, and 888 training interactions, serving as a high-density "
        "testbed for LLM semantic auditing under controlled API costs.",
        bold_prefix="Amazon Pipeline: "
    )
    add_paragraph(
        doc,
        "3. Yelp Academic Loader (yelp_loader.py): Ingests Yelp local business reviews, parsing business metadata, star ratings, timestamps, and customer reviews. "
        "A 5-core filter produces |U| = 70 users, |I| = 100 items, and 1,152 training interactions.",
        bold_prefix="Yelp Pipeline: "
    )

    # 4.4 Neural Backbones
    add_section_heading(doc, "4.4 Graph Neural Network Backbones and Contrastive Engine")
    add_paragraph(
        doc,
        "The neural modeling layer abstracts model architectures behind a unified interface supporting forward prediction, embedding extraction, and contrastive view generation:"
    )
    add_bullet(doc, "NeuMF (neumf.py): Implements dual towers—Generalized Matrix Factorization (element-wise product of user-item embeddings) and Multi-Layer Perceptron (3-layer feed-forward network with ReLU). The tower outputs are concatenated and projected to clean rating logits z(u, i) in R^K.", bold_prefix="NeuMF Backbone: ")
    add_bullet(doc, "SimGCL (simgcl.py): Builds upon LightGCN message passing. In each forward training pass, two perturbed contrastive views are generated by injecting scaled uniform noise into normalized embeddings: z' = Normalize(e + epsilon * sgn(e) * Delta). This eliminates graph reconstruction overhead while maximizing hyperspherical uniformity.", bold_prefix="SimGCL Backbone: ")
    add_bullet(doc, "SGL (sgl.py): Implements structural edge dropout on the normalized adjacency matrix: A_hat' = (M' odot A) * D'^(-1) with dropout ratio p_drop = 0.10. Contrastive views are generated by passing node embeddings through distinct augmented adjacency graphs.", bold_prefix="SGL Backbone: ")

    # 4.5 MIRF Filter & BLAS Acceleration
    add_section_heading(doc, "4.5 MIRF Filter Modules and High-Performance BLAS Acceleration")
    add_paragraph(
        doc,
        "The MIRF ensemble synthesizes heterogeneous signals across three specialized engines:"
    )
    add_paragraph(
        doc,
        "Review-Bombing Vectorization (bomb_scorer.py): Rolling interaction velocity and rating polarity skew are computed simultaneously via 2D BLAS array operations. "
        "Given interaction array X of shape (N, 3) containing [user_id, item_id, timestamp], items are pre-binned into temporal windows. The score matrix is computed as: "
        "Score_Matrix = W @ Feature_Tensor, executed entirely through hardware-optimized BLAS Level-3 GEMM routines. Benchmarking on 42,000 interactions demonstrates "
        "an execution latency of exactly 4.95 milliseconds, representing a >500x speedup over nested Python loops.",
        bold_prefix="Vectorized Burst Scoring: "
    )
    add_paragraph(
        doc,
        "Thread-Safe SQLite WAL Caching (cache.py): LLM review auditing queries are cryptographically hashed using SHA-256 over the tuple (user_id, item_id, rating, review_text). "
        "The cache database is initialized with PRAGMA journal_mode = WAL and PRAGMA synchronous = NORMAL. This enables concurrent multi-threaded worker reads while background "
        "threads write audit results without database lock contention. Cache hits execute in O(1) time (< 0.1 ms), entirely insulating training runs from network latency.",
        bold_prefix="SQLite WAL Caching: "
    )

    # 4.6 Denoised SUPER Engine
    add_section_heading(doc, "4.6 Denoised SUPER Calibrated Recommender Engine")
    add_paragraph(
        doc,
        "The SUPER engine implementation guarantees calibrated item exposure while actively purging malicious interactions:"
    )
    add_numbered(doc, "1.", "Pareto Partitioning (pareto_partition.py): Sorts all catalog items by interaction frequency in descending order. The top items accounting for cumulative interaction fraction rho_pareto = 0.20 are assigned to Head catalog I_head; the remaining items form Tail catalog I_tail.", bold_prefix="Partitioning: ")
    add_numbered(doc, "2.", "Blueprint Purification (blueprint.py): For user u, observed interaction edges are filtered against trust threshold theta_trust = 0.50. Edges with w_ui < theta_trust (malicious bot interactions) are expunged from blueprint B_u.", bold_prefix="Purification: ")
    add_numbered(doc, "3.", "Calibrated Inclination (inclination.py): Calculates personalized tail preference alpha_u using reliability-weighted sums, smoothed by epsilon_smooth = 1e-6 to prevent zero division on sparse profiles.", bold_prefix="Inclination: ")
    add_numbered(doc, "4.", "Trajectory Interleaving (merger.py): Greedily selects candidate items from Head and Tail queues to maintain cumulative tail proportion <= alpha_u at every rank k in {1, ..., K}.", bold_prefix="Interleaving: ")

    # 4.7 Pseudocode
    add_section_heading(doc, "4.7 Algorithmic Workflows and End-to-End Pseudocode")
    add_paragraph(
        doc,
        "To provide exact algorithmic specifications, Algorithms 4.1 and 4.2 document the procedural logic of MIRF reliability estimation "
        "and end-to-end RRFN-LLM-SUPER training and inference."
    )

    alg1_code = (
        "Algorithm 4.1: MIRF Multi-View Sample Reliability Estimation\n"
        "Input:  Interaction edge (u, i, r_tilde, S, t), model predictions p(u, i), transition matrix T,\n"
        "        weights alpha, beta, gamma, rolling density parameters mu_D, sigma_D, threshold theta_bomb\n"
        "Output: Sample reliability weight w_ui in [0, 1]\n"
        "----------------------------------------------------------------------------------------------------\n"
        " 1: E_rating <- sum_{k=1}^K k * p_k(u, i)                          // Expected clean rating\n"
        " 2: R_RRFN   <- exp( - (r_tilde - E_rating)^2 / (2 * sigma_RRFN^2) ) // Statistical consistency\n"
        " 3: if review text S is available and beta > 0 then\n"
        " 4:     query_hash <- SHA256(u, i, r_tilde, S)\n"
        " 5:     if query_hash in SQLite_WAL_Cache then\n"
        " 6:         R_LLM <- SQLite_WAL_Cache.get(query_hash)\n"
        " 7:     else\n"
        " 8:         R_LLM <- LLM_Audit_Prompt(S, r_tilde)              // Asynchronous API audit\n"
        " 9:         SQLite_WAL_Cache.put(query_hash, R_LLM)\n"
        "10:     end if\n"
        "11: else\n"
        "12:     R_LLM <- 1.0; beta <- 0.0; renormalize(alpha, gamma)    // Modality fallback\n"
        "13: end if\n"
        "14: Z_temp   <- (D_recent(i, t) - mu_D(i)) / (sigma_D(i) + epsilon)   // 2D BLAS acceleration\n"
        "15: S_pol    <- |N_1(i, t) - N_5(i, t)| / (N_1(i, t) + N_5(i, t) + epsilon) // Polarity skew\n"
        "16: R_bomb   <- 1.0 - Sigmoid( omega_1 * Z_temp + omega_2 * S_pol - theta_bomb )\n"
        "17: w_ui     <- alpha * R_RRFN + beta * R_LLM + gamma * R_bomb     // Fused convex combination\n"
        "18: return clamp(w_ui, 0.0, 1.0)"
    )
    add_code_block(doc, alg1_code, caption="Algorithm 4.1: MIRF Multi-View Sample Reliability Estimation")

    alg2_code = (
        "Algorithm 4.2: End-to-End RRFN-LLM-SUPER Training and Calibrated Inference\n"
        "Input:  Graph G = (U, I, E_tilde), anchor set A_clean, epochs E, batch size B, Pareto ratio rho=0.20\n"
        "Output: Trained Head model M_pop, Tail model M_tail, top-K recommendations for all users u in U\n"
        "----------------------------------------------------------------------------------------------------\n"
        " 1: Estimate anchor transition prior T_hat from A_clean; compute T_lambda^dagger = (T^T T + lambda I)^-1 T^T\n"
        " 2: Partition catalog I into Head I_head and Tail I_tail at Pareto cut-off rho\n"
        " 3: For each interaction edge (u, i) in E_tilde: compute reliability weight w_ui via Algorithm 4.1\n"
        " 4: for epoch = 1 to E do\n"
        " 5:     Sample mini-batch B_head subset of E_head, B_tail subset of E_tail\n"
        " 6:     Compute forward surrogate risk loss L_risk and risk-weighted InfoNCE L_cl for M_pop and M_tail\n"
        " 7:     Backpropagate joint losses L_total and update parameters theta_head, theta_tail via AdamW\n"
        " 8: end for\n"
        " 9: // Inference & Recommendation Assembly:\n"
        "10: for each user u in U do\n"
        "11:     Purify blueprint B_u = { i in I | (u, i) in E_tilde and w_ui >= theta_trust }\n"
        "12:     Compute calibrated tail inclination alpha_u = sum_{i in B_u intersect I_tail} w_ui / (sum w_ui + eps)\n"
        "13:     Generate candidate queues C_head = M_pop.predict(u) \\ B_u, C_tail = M_tail.predict(u) \\ B_u\n"
        "14:     Assemble top-K list via Rank-Calibrated Trajectory Interleaving tracking alpha_u\n"
        "15: end for\n"
        "16: return Top-K recommendations"
    )
    add_code_block(doc, alg2_code, caption="Algorithm 4.2: End-to-End RRFN-LLM-SUPER Training and Calibrated Inference")

    # 4.8 Streamlit Dashboard
    add_section_heading(doc, "4.8 Interactive Telemetry and Streamlit Web Dashboard")
    add_paragraph(
        doc,
        "To enable real-time observability, hyperparameter sensitivity analysis, and audit inspection, we implemented an interactive 4-panel "
        "Streamlit web dashboard (dashboard.py). As illustrated in Figure 4.1, the dashboard provides live telemetry across four specialized views:"
    )
    add_bullet(doc, "Tab 1 (Multi-View Signal Fusion): Displays live radar telemetry and provides interactive sliders to adjust signal weights (alpha, beta, gamma), showing immediate recalculation of reliability distributions.", bold_prefix="Tab 1: ")
    add_bullet(doc, "Tab 2 (Noise Transition Matrix & Inversion Heatmaps): Renders interactive heatmaps of estimated transition matrix T and Tikhonov regularized inverse T_lambda^dagger, displaying condition number kappa(T) and operator norm ||T_lambda^dagger||_2.", bold_prefix="Tab 2: ")
    add_bullet(doc, "Tab 3 (Catalog Popularity Exposure & Trade-Offs): Plots popularity decile distributions, comparing raw LTC vs. genuine g-LTC and tracking Spam Target Exposure Rate (STER@10).", bold_prefix="Tab 3: ")
    add_bullet(doc, "Tab 4 (Live User Inspector & Audit Explorer): Allows engineers to select any individual user, inspect their historical blueprint, verify which edges were purged by MIRF, and trace top-K recommendation assembly.", bold_prefix="Tab 4: ")

    add_figure(
        doc,
        r"paper/figures/fig8_dashboard_multipanel.png",
        caption_text="Four-Panel Interactive Streamlit Telemetry Dashboard: (a) Multi-View Signal Fusion Radar, (b) Noise Transition Matrix and Tikhonov Inversion Heatmaps, (c) Catalog Exposure and STER@10 Telemetry, and (d) Live User Inspector and Recommendation Auditor.",
        fig_num_str="4.1",
        width_cm=14.4
    )

    # 4.9 Software Hardening
    add_section_heading(doc, "4.9 Software Hardening, Numerical Protection, and Concurrency Controls")
    add_paragraph(
        doc,
        "To guarantee fail-safe stability in production and testing environments, the codebase includes four key software hardening protections:"
    )
    add_numbered(doc, "1. Tikhonov Regularized Pseudo-Inversion:", "All matrix inversion operations in transition_matrix.py utilize T_lambda^dagger = (T^T T + lambda I)^(-1) T^T with damping factor lambda = 1e-4. This strictly prevents singular matrix crashes even when rating classes have zero empirical support.", bold_prefix="Numerical Inversion Guard: ")
    add_numbered(doc, "2. Log-Variance Clamping in Variational Backbones:", "In vaecf.py, latent log-variance tensors are clamped to [-10.0, 10.0] prior to exponential exponentiation, preventing numerical underflow to 0.0 or overflow to infinity.", bold_prefix="Float32 Clamping: ")
    add_numbered(doc, "3. Zero-Division and Empty-Candidate Guards:", "In merger.py and inclination.py, user blueprints with zero tail interactions are protected by epsilon_smooth = 1e-6. If a candidate queue becomes exhausted before top-K assembly completes, the interleaver dynamically falls back to the alternate partition queue.", bold_prefix="Empty Set Guards: ")
    add_numbered(doc, "4. Secret Redaction and API Sanitization:", "In auditor.py and cache.py, OpenAI API keys and environment variables are sanitized; credentials are never written to logs, console telemetry, or serialized model checkpoints.", bold_prefix="Security Hardening: ")

    doc.add_page_break()
