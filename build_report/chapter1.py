"""
Chapter 1: Introduction and Motivation
Comprehensive, publication-grade text for BCSE497J Project Report.
"""

from report_builder import (
    add_chapter_heading, add_section_heading, add_subsection_heading,
    add_subsubsection_heading, add_paragraph, add_bullet, add_numbered,
    add_equation, add_callout_box
)

def build_chapter_1(doc):
    add_chapter_heading(doc, "CHAPTER 1", "INTRODUCTION AND MOTIVATION")

    # 1.1 Background and Motivation
    add_section_heading(doc, "1.1 Background and Motivation")
    
    add_paragraph(
        doc,
        "In the contemporary digital economy, web-scale recommender systems have transitioned from optional convenience features "
        "into mission-critical computational backbones. From massive e-commerce marketplaces such as Amazon and Alibaba to digital media "
        "streaming services like Netflix and Spotify, and social networking hubs like YouTube and Yelp, personalized recommendation algorithms "
        "curate the daily digital experiences of billions of users worldwide. These algorithmic engines process vast volumes of user behavioral feedback "
        "to predict user preferences, drive user engagement, maximize conversion rates, and generate billions of dollars in Gross Merchandise Value (GMV). "
        "At their core, modern recommendation systems conceptualize the problem of personalization through the mathematical formalism of bipartite graphs."
    )
    
    add_paragraph(
        doc,
        "Formally, let G = (U, I, E) represent a bipartite user-item interaction graph, where U = {u_1, u_2, ..., u_|U|} denotes the set of unique users, "
        "I = {i_1, i_2, ..., i_|I|} denotes the set of unique catalog items, and E subset of U x I denotes the observed set of behavioral edges or interactions. "
        "In implicit feedback environments, an observed edge (u, i) in E signifies a click, view, purchase, or bookmark. In explicit feedback settings, "
        "each interaction edge is accompanied by a discrete ordinal rating r_ui in {1, 2, ..., K} (typically representing a 1-to-5 star rating) and, "
        "in rich multi-modal platforms, a corresponding natural language review text S_ui and an interaction timestamp t_ui."
    )
    
    add_paragraph(
        doc,
        "Over the past decade, Graph Neural Networks (GNNs)—most notably Graph Convolutional Networks (GCNs) and simplified variants such as LightGCN—"
        "have emerged as the de facto state of the art for collaborative filtering. By recursively performing message passing and neighborhood aggregation "
        "across the bipartite graph topology, GNNs explicitly capture high-order collaborative signals that transcend simple pairwise similarity. "
        "However, when deployed in commercial environments, graph-based recommender systems encounter two profound, orthogonal, and systemic vulnerabilities "
        "that severely degrade their robustness, fairness, and utility: adversarial data poisoning and structural catalog popularity bias."
    )

    # 1.2 Systemic Vulnerabilities
    add_section_heading(doc, "1.2 Systemic Vulnerabilities in Graph Recommenders")
    
    add_subsection_heading(doc, "1.2.1 Adversarial Data Poisoning and Shilling Attacks")
    add_paragraph(
        doc,
        "The first existential threat stems from the open, democratized nature of modern web platforms, where any registered entity can inject ratings "
        "and reviews. Malicious actors, competing commercial vendors, and automated bot networks frequently execute coordinated adversarial data injection "
        "campaigns, commonly categorized as shilling attacks. The primary economic objective of a shilling campaign is typically a 'push attack' "
        "(artificially inflating the visibility, recommendation frequency, and ranking of designated target items) or a 'nuke attack' "
        "(deliberately sabotaging a competitor's products through coordinated low ratings)."
    )
    add_paragraph(
        doc,
        "Among shilling attack strategies, the bandwagon shilling attack is exceptionally insidious and prevalent. In a bandwagon campaign, automated bot "
        "accounts inject extreme positive feedback (5-star ratings) onto low-popularity target items to promote them, while concurrently forging positive "
        "interactions on top popular 'anchor' items in the catalog. By injecting interactions with widely consumed, highly rated items, the malicious bot "
        "profiles blend seamlessly into the authentic user population, effectively camouflaging their fraudulent activity against naive variance-based filters. "
        "Furthermore, in modern platforms, coordinated attacks manifest as temporal review-bombing bursts, where hundreds of synthetic accounts flood "
        "a specific product with polarizing ratings within narrow time windows, accompanied by template-generated bot text reviews."
    )
    add_paragraph(
        doc,
        "Because standard collaborative filtering models and GNNs operate under the fundamental assumption that all observed graph edges E are authentic "
        "and clean, malicious interactions are propagated through high-order graph convolution layers. Consequently, the embedding representations of genuine "
        "users and items become severely corrupted, causing spam items to pollute top-K recommendations across the platform."
    )

    add_subsection_heading(doc, "1.2.2 Structural Catalog Popularity Bias and Long-Tail Starvation")
    add_paragraph(
        doc,
        "The second systemic vulnerability arises from the intrinsic power-law (Pareto) distribution that governs human interaction behavior. "
        "In virtually every real-world e-commerce and media catalog, approximately 20% of the items (the 'head' or popular items) attract over 80% "
        "of all interactions, while the remaining 80% of items (the 'tail' or niche catalog) compete for a minuscule fraction of user engagement. "
        "Standard machine learning objectives optimize global empirical loss functions such as Cross-Entropy or Bayesian Personalized Ranking (BPR). "
        "Under these objectives, models maximize overall accuracy by repeatedly recommending head items, which naturally minimizes empirical penalty "
        "due to their high frequency in training data."
    )
    add_paragraph(
        doc,
        "This feedback loop leads to the Matthew Effect ('the rich get richer, and the poor get poorer'), severely starving relevant long-tail items "
        "of organic exposure. From a business perspective, catalog starvation creates serious platform vulnerabilities: niche creators and small merchants "
        "churn due to lack of visibility, catalog diversity plummets, and users suffer from monotonous, repetitive recommendations that fail to reflect "
        "their personalized taste for niche discoveries."
    )

    add_subsection_heading(doc, "1.2.3 The Fatal Concurrence: Why Isolated Solutions Collapse")
    add_paragraph(
        doc,
        "Prior academic literature has treated adversarial shilling defense and popularity debiasing as entirely disconnected research tracks. "
        "Denoising researchers design heuristic edge pruning or statistical loss corrections assuming uniform, unbiased interaction graphs. "
        "Conversely, debiasing researchers design popularity reweighting or catalog partition frameworks (such as standalone SUPER) under the idealized "
        "premise that the underlying training graph is completely clean and uncorrupted."
    )
    add_paragraph(
        doc,
        "When both phenomena co-occur in real-world platforms, isolated solutions collapse catastrophically. Specifically, when a bandwagon attack is "
        "executed on a popularity-biased graph, attackers deliberately select head items to camouflage their profiles and push obscure tail items. "
        "A standalone debiasing algorithm (such as SUPER), observing high user interaction with both head and pushed tail items, mistakenly treats the "
        "injected spam interactions as legitimate user enthusiasm for long-tail discovery! Consequently, standard debiasing models aggressively allocate "
        "long-tail quotas to the attacker's spam target items, amplifying the attack's economic damage while reporting falsely elevated 'long-tail coverage'."
    )

    # 1.3 Problem Statement
    add_section_heading(doc, "1.3 Problem Statement and Formal Research Questions")
    add_paragraph(
        doc,
        "Given a partially observed, potentially poisoned bipartite user-item graph G_tilde = (U, I, E_tilde), where E_tilde contains an unobserved mixture "
        "of authentic clean interactions E* and malicious adversarial injection edges E_attack, and where item interaction frequencies exhibit severe "
        "Pareto skew, how can an end-to-end recommender system simultaneously:"
    )
    add_bullet(doc, "1. Identify and downweight corrupted interaction edges without discarding authentic user feedback;", bold_prefix="Denoising: ")
    add_bullet(doc, "2. Provide rigorous mathematical guarantees against gradient explosion and noise amplification during model optimization;", bold_prefix="Theoretical Soundness: ")
    add_bullet(doc, "3. Prevent poisoned interactions from distorting self-supervised graph contrastive representation embeddings;", bold_prefix="Graph Contrastive Alignment: ")
    add_bullet(doc, "4. Calibrate top-K recommendations to match individual users' genuine historical tail preferences without pushing malicious spam targets.", bold_prefix="Calibrated Debiasing: ")
    
    add_paragraph(
        doc,
        "To guide the engineering and scientific investigation in this project report, we formulate four formal Research Questions (RQs):"
    )
    add_numbered(doc, "RQ1:", "How can stochastic rating noise transitions be formally inverted such that parameter gradient updates remain strictly bounded, non-singular, and risk-consistent with respect to the clean Bayes distribution?", bold_prefix="Noise Inversion & Risk Consistency: ")
    add_numbered(doc, "RQ2:", "Can multi-modal signals spanning neural prediction discrepancies, LLM semantic review audits, and temporal-polarity burst patterns be fused into an ultra-low latency interaction reliability filter operating at sub-5-millisecond execution times?", bold_prefix="Multi-View Reliability Auditing: ")
    add_numbered(doc, "RQ3:", "How does weighting InfoNCE contrastive alignment losses by sample-level reliability weights prevent representation collapse and graph embedding pollution under coordinated bandwagon attacks?", bold_prefix="Risk-Aware Graph Contrastive Learning: ")
    add_numbered(doc, "RQ4:", "How can historical user blueprints be purified to separate genuine long-tail interest from synthetic spam promotion, and how do we formally measure authentic catalog recovery versus deceptive spam exposure?", bold_prefix="Purified Calibrated Exposure: ")

    # 1.4 Objectives
    add_section_heading(doc, "1.4 Objectives of the Project")
    add_paragraph(
        doc,
        "In direct alignment with the course outcomes (CO4) and the academic criteria prescribed for BCSE497J Project I, this project accomplishes "
        "the following five specific, measurable, and technically rigorous objectives:"
    )
    add_numbered(doc, "Objective 1:", "Derive formal mathematical foundations for noise transition modeling, establishing a proven spectral operator norm upper bound for Tikhonov-regularized noise inversion (||T_lambda^dagger||_2 <= 1 / (2*sqrt(lambda))) to guarantee Lipschitz gradient stability, and establish asymptotic excess risk consistency bounds O(delta + 1/sqrt(N)).", bold_prefix="Theoretical Formulations: ")
    add_numbered(doc, "Objective 2:", "Architect and implement the Multi-view Interaction Reliability Filter (MIRF), integrating statistical transition divergence, asynchronous Large Language Model (LLM) review text verification with a thread-safe SQLite WAL caching layer, and 2D BLAS vectorized review-bombing burst detection achieving >500x execution speedup.", bold_prefix="MIRF Filter Engineering: ")
    add_numbered(doc, "Objective 3:", "Integrate risk-aware Graph Contrastive Learning (GCL) across modern neural backbones (SimGCL with latent uniform perturbations and SGL with structural edge dropout), modulating contrastive alignment through sample reliability weights w_ui.", bold_prefix="Robust Representation Learning: ")
    add_numbered(doc, "Objective 4:", "Design the Denoised SUPER Recommender Engine with purified user interaction blueprints and rank-calibrated trajectory interleaving, formulating attack-aware evaluation metrics: Genuine Long-Tail Coverage (g-LTC@K) and Spam Target Exposure Rate (STER@K).", bold_prefix="Debiased Engine & Metrics: ")
    add_numbered(doc, "Objective 5:", "Execute comprehensive 5-seed empirical benchmark evaluations across three diverse domains (MovieLens-1M sub-graph, Amazon Electronics, Yelp Academic), conducting paired Student's t-tests, full component ablations, and a 1,000-sample Monte Carlo economic simulation.", bold_prefix="Empirical & Economic Validation: ")

    # 1.5 Grounding Key Concepts: SUP, RRFN, and MIRF
    add_section_heading(doc, "1.5 Grounding Key Concepts: SUP, RRFN, and MIRF")
    add_paragraph(
        doc,
        "To establish absolute terminological clarity and conceptual grounding throughout this report, we formally define the core components "
        "that constitute our unified RRFN-LLM-SUPER framework:"
    )
    
    add_subsection_heading(doc, "1.5.1 What is SUP / SUPER?")
    add_paragraph(
        doc,
        "In recommender systems literature, SUP (also referred to as SUPER) stands for Soft Blueprint Merging for Calibrated Long-Tail Recommendation (Yang et al., 2022). "
        "SUPER addresses catalog popularity skew by partitioning the item catalog into popular 'Head' items (M_pop) and niche 'Tail' items (M_tail). "
        "During recommendation inference, SUPER calculates each individual user's historical tail preference inclination alpha_u in [0, 1] based on their past interaction 'blueprint'. "
        "It then assembles a top-K recommendation list by soft-merging candidates from M_pop and M_tail to ensure the recommended list's popularity distribution "
        "closely matches the user's historical profile."
    )
    add_paragraph(
        doc,
        "Crucial Vulnerability of Standalone SUP: Standalone SUPER inherently presupposes that the user's historical interaction blueprint is clean and honest. "
        "When an adversarial bot injects fake interactions, SUPER faithfully ingests the poisoned edges into the blueprint, skewing alpha_u and elevating attacked "
        "spam items into the user's calibrated quota."
    )

    add_subsection_heading(doc, "1.5.2 What is RRFN?")
    add_paragraph(
        doc,
        "RRFN stands for the Robust Risk-consistent Fusion Network. RRFN is a noise-transition modeling framework that acknowledges that observed ratings "
        "Y_tilde are noisy realizations of unobserved ground-truth clean ratings Y*. RRFN identifies high-confidence anchor items in the catalog to estimate "
        "a K x K stochastic rating transition matrix T_ij = P(Y_tilde = j | Y* = i). By inverting this matrix using Tikhonov regularization and optimizing a forward-corrected "
        "surrogate cross-entropy loss L_risk, RRFN guarantees that model optimization asymptotically converges toward the clean Bayes-optimal risk."
    )

    add_subsection_heading(doc, "1.5.3 What is MIRF?")
    add_paragraph(
        doc,
        "MIRF stands for the Multi-view Interaction Reliability Filter. MIRF is our novel tri-view ensemble layer that audits the authenticity of every graph edge (u, i) "
        "by fusing three complementary signal views:"
    )
    add_bullet(doc, "Statistical Model Prediction Consistency (R_RRFN): Measures whether the observed rating aligns with the model's expected rating under the estimated transition matrix T.", bold_prefix="View 1: ")
    add_bullet(doc, "Semantic Review Text Auditing (R_LLM): Analyzes natural language review text S_ui using LLM prompts to detect sentiment polarity conflicts and bot syntax, cached in a thread-safe SQLite WAL database.", bold_prefix="View 2: ")
    add_bullet(doc, "Vectorized Review-Bombing Burst Detection (R_bomb): Simultaneously scores sudden temporal acceleration (Z_temp) and polarity skew (S_pol) via 2D BLAS matrix operations in sub-5ms runtime.", bold_prefix="View 3: ")
    add_paragraph(
        doc,
        "MIRF synthesizes these three views into a unified continuous sample reliability weight w_ui in [0, 1] for every interaction edge."
    )

    add_subsection_heading(doc, "1.5.4 Our Unified Framework: RRFN-LLM-SUPER")
    add_paragraph(
        doc,
        "RRFN-LLM-SUPER is the integrated end-to-end framework combining all four pillars. It uses MIRF to audit graph interactions, applies Tikhonov-regularized "
        "noise inversion for risk-consistent loss computation, weights self-supervised Graph Contrastive Learning (SimGCL/SGL) by sample trust weights, "
        "and purifies SUPER blueprints so that calibrated top-K recommendations reflect authentic long-tail discovery while repelling adversarial shilling."
    )

    # 1.6 Scope and Boundaries
    add_section_heading(doc, "1.6 Scope and Boundaries of the Work")
    add_paragraph(
        doc,
        "The scope of this project encompasses the complete lifecycle of designing, mathematically proving, implementing, and benchmarking "
        "the RRFN-LLM-SUPER framework. Specifically, the boundaries and operational assumptions of this work are defined as follows:"
    )
    add_bullet(doc, "Benchmark Datasets: Rigorous evaluation across three multi-modal domains—MovieLens-1M sub-graph (ratings, genres, timestamps; 300 users, 3,405 items), Amazon Electronics (ratings, text reviews, timestamps; 60 users, 81 items), and Yelp Academic (ratings, text reviews, business metadata; 70 users, 100 items).", bold_prefix="Data Modalities: ")
    add_bullet(doc, "Adversarial Attack Scenarios: Evaluation under 15% bot account injection executing Bandwagon Shilling Attacks, Random Uniform Rating Noise, and Coordinated Temporal Review-Bombing Nuke Attacks.", bold_prefix="Threat Models: ")
    add_bullet(doc, "Neural Architectures: Support for deep collaborative filtering (NeuMF) and state-of-the-art graph contrastive architectures (SimGCL with latent uniform perturbations and SGL with graph edge dropout).", bold_prefix="Backbones: ")
    add_bullet(doc, "Statistical Rigor: All empirical benchmarks are conducted across S = 5 independent random seeds (42 through 46), reporting sample mean, sample standard deviation, and paired Student's t-test statistical significance (p < 0.05).", bold_prefix="Multi-Seed Evaluation: ")
    add_bullet(doc, "Out-of-Scope Constraints: Real-time distributed online streaming updates (e.g., Apache Kafka / Flink streaming graph ingestion) and live production A/B traffic testing are excluded due to computing infrastructure constraints. Offline Monte Carlo bootstrap simulation is utilized for economic modeling.", bold_prefix="System Boundaries: ")

    # 1.7 Organization of Report
    add_section_heading(doc, "1.7 Organization of the Project Report")
    add_paragraph(
        doc,
        "The remainder of this project report is systematically organized into the following chapters:"
    )
    add_bullet(doc, "Chapter 2 (Literature Review and Theoretical Background): Conducts an exhaustive survey of prior work in collaborative filtering, adversarial shilling attacks, noise transition modeling, popularity debiasing, graph contrastive learning, and LLM auditing, concluding with an explicit literature positioning matrix.", bold_prefix="Chapter 2: ")
    add_bullet(doc, "Chapter 3 (System Design and Mathematical Methodology): Details the complete theoretical architecture of RRFN-LLM-SUPER, presents formal proofs for Theorem 1 (operator norm bound) and Theorem 2 (excess clean risk consistency), details the MIRF pipeline, and formalizes all evaluation metrics.", bold_prefix="Chapter 3: ")
    add_bullet(doc, "Chapter 4 (System Implementation and Architecture Details): Provides a deep software engineering walkthrough of all 35+ source modules across data loaders, neural models, BLAS acceleration, SQLite WAL caching, pseudocode listings, and the Streamlit telemetry dashboard.", bold_prefix="Chapter 4: ")
    add_bullet(doc, "Chapter 5 (Experimental Evaluation, Results, and Discussion): Reports the 5-seed empirical benchmark results across MovieLens-1M, Amazon, and Yelp, analyzes detection metrics, presents component ablations, loss curves, and economic simulations, and discusses engineering trade-offs honestly.", bold_prefix="Chapter 5: ")
    add_bullet(doc, "Chapter 6 (Conclusion and Future Work): Summarizes key contributions, synthesizes research findings, addresses realistic system limitations, and outlines concrete avenues for future research.", bold_prefix="Chapter 6: ")
    add_bullet(doc, "References & Appendices: Provides 28+ references in strict APA format, repository directory trees, hyperparameter tables, code listings, and plagiarism compliance certification.", bold_prefix="Back Matter: ")

    doc.add_page_break()
