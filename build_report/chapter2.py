"""
Chapter 2: Literature Review and Theoretical Background
Comprehensive, publication-grade text for BCSE497J Project Report.
"""

from report_builder import (
    add_chapter_heading, add_section_heading, add_subsection_heading,
    add_paragraph, add_bullet, add_numbered, add_equation, add_table_custom
)

def build_chapter_2(doc):
    add_chapter_heading(doc, "CHAPTER 2", "LITERATURE REVIEW AND THEORETICAL BACKGROUND")

    # 2.1 Overview of Collaborative Filtering
    add_section_heading(doc, "2.1 Overview of Collaborative Filtering Paradigms")
    add_paragraph(
        doc,
        "Collaborative Filtering (CF) constitutes the foundational cornerstone of modern recommender systems. The fundamental hypothesis "
        "underlying collaborative filtering posits that users who shared similar behavioral preferences in the past will exhibit congruent tastes "
        "in the future. Early collaborative filtering algorithms relied on neighborhood-based heuristics, including User-Based CF and Item-Based CF, "
        "which computed pairwise similarity metrics such as Pearson Correlation Coefficient or Cosine Similarity over sparse interaction matrices."
    )
    add_paragraph(
        doc,
        "During the Netflix Prize competition (2006–2009), latent factor models based on Matrix Factorization (MF) demonstrated decisive superiority "
        "over neighborhood methods (Koren et al., 2009). Matrix factorization projects users and items into a shared low-dimensional latent embedding space "
        "$\\mathbb{R}^d$, approximating the observed rating $r_{ui}$ through the inner product of user latent vector $\\mathbf{p}_u$ and item latent vector $\\mathbf{q}_i$:"
    )
    add_equation(doc, "\\hat{r}_{ui} = \\mu + b_u + b_i + \\mathbf{p}_u^T \\mathbf{q}_i", "(2.1)")
    add_paragraph(
        doc,
        "where $\\mu$ represents the global bias, and $b_u, b_i$ denote the user-specific and item-specific bias parameters respectively. "
        "For implicit feedback datasets, Rendle et al. (2009) formulated Bayesian Personalized Ranking (BPR), a pairwise ranking optimization criterion "
        "that maximizes the posterior margin between observed positive items and unobserved negative items."
    )
    add_paragraph(
        doc,
        "With the deep learning revolution, Neural Collaborative Filtering (NCF / NeuMF) proposed by He et al. (2017) integrated linear Generalized "
        "Matrix Factorization (GMF) with non-linear Multi-Layer Perceptrons (MLP) to capture complex, non-linear user-item interaction dynamics. "
        "More recently, Graph Neural Networks (GNNs) transformed collaborative filtering by explicitly representing the user-item interaction matrix as "
        "a bipartite graph $\\mathcal{G} = (\\mathcal{U}, \\mathcal{I}, \\mathcal{E})$. Neural Graph Collaborative Filtering (NGCF; Wang et al., 2019) introduced high-order feature propagation across graph "
        "neighborhoods. LightGCN (He et al., 2020) subsequently demonstrated that non-linear feature transformations and self-connections in standard GCNs "
        "are redundant for collaborative filtering, proposing a streamlined architecture based solely on normalized neighborhood smoothing:"
    )
    add_equation(doc, "\\mathbf{e}_u^{(l+1)} = \\sum_{i \\in \\mathcal{N}_u} \\frac{1}{\\sqrt{|\\mathcal{N}_u| \\cdot |\\mathcal{N}_i|}} \\mathbf{e}_i^{(l)}", "(2.2)")
    add_paragraph(
        doc,
        "While LightGCN achieves exceptional recommendation accuracy, its high-order message passing inherently acts as a noise amplifier: any malicious or poisoned edge injected into the graph is recursively diffused across "
        "multiple hops, contaminating embeddings throughout the entire bipartite structure."
    )

    # 2.2 Adversarial Shilling Attacks
    add_section_heading(doc, "2.2 Adversarial Shilling Attacks in Recommender Systems")
    add_paragraph(
        doc,
        "As recommender systems increasingly dictate commercial transaction volumes, they have become lucrative targets for adversarial manipulation. "
        "Adversarial data poisoning in recommendation—commonly designated as shilling attacks—involves injecting synthetic bot user profiles containing "
        "carefully structured interaction patterns to manipulate recommendation outputs (Mobasher et al., 2007; Si et al., 2022; Wu et al., 2021)."
    )
    add_paragraph(
        doc,
        "Shilling attack strategies are categorized based on the composition of items within the attacker's synthetic user profiles:"
    )
    add_bullet(doc, "Random Attacks: Bot profiles inject maximal ratings on target items and assign randomly sampled ratings to a uniformly chosen subset of filler items. While simple to implement, random attacks are readily identified by elementary variance detection algorithms.", bold_prefix="1. Random Attacks: ")
    add_bullet(doc, "Average Attacks: Filler item ratings are sampled from each item's empirical catalog average. This substantially increases detection difficulty compared to random noise.", bold_prefix="2. Average Attacks: ")
    add_bullet(doc, "Bandwagon Shilling Attacks: The attacker selects a subset of highly popular catalog items ('anchor items') and injects high ratings on these popular items alongside extreme positive ratings on the obscure target items. Because the bot profiles share strong co-occurrence with genuine mainstream user behavior on popular items, the bot profiles appear highly authentic, evading standard anomaly detectors and establishing strong collaborative filtering links between popular items and target items.", bold_prefix="3. Bandwagon Attacks: ")
    add_bullet(doc, "Review-Bombing Nuke Attacks: Coordinated negative campaigns where numerous bot accounts concentrate extreme negative ratings (1-star) and defamatory text reviews on a competitor's item within a concentrated temporal window, driving its rating down and destroying its algorithmic visibility.", bold_prefix="4. Review-Bombing Attacks: ")
    add_paragraph(
        doc,
        "Early defense literature focused on heuristic detection filters, including statistical user clustering, spectral graph partitioning, and "
        "degree-based outlier rejection (Chirita et al., 2005; Mehta et al., 2007). However, these static defense mechanisms fail against sophisticated "
        "bandwagon profiles that mimic genuine rating distributions, necessitating principled, loss-level and graph-level defense mechanisms."
    )

    # 2.3 Noise-Transition Modeling
    add_section_heading(doc, "2.3 Noise-Transition Modeling and Loss Correction")
    add_paragraph(
        doc,
        "In statistical machine learning, learning with noisy labels has been extensively studied within supervised computer vision (Patrini et al., 2017; "
        "Natarajan et al., 2013). The central premise models observed labels $\\tilde{Y}$ as probabilistic corruptions of true clean labels $Y^*$ governed by "
        "a class-conditional transition matrix $T \\in \\mathbb{R}^{K \\times K}$:"
    )
    add_equation(doc, "T_{ij} = P(\\tilde{Y} = j \\mid Y^* = i), \\quad \\text{subject to } \\sum_{j=1}^K T_{ij} = 1, \\quad \\forall i \\in \\{1, \\dots, K\\}", "(2.3)")
    add_paragraph(
        doc,
        "Loss correction techniques generally follow two paradigms: Backward Correction, which multiplies the empirical loss vector by the inverse transition matrix:"
    )
    add_equation(doc, "\\ell_{\\mathrm{backward}} = T^{-1} \\ell(\\mathbf{p}, \\tilde{Y})", "(2.4)")
    add_paragraph(
        doc,
        "and Forward Correction, which projects model prediction probabilities through the transition matrix:"
    )
    add_equation(doc, "\\ell_{\\mathrm{forward}} = \\ell_{\\mathrm{CE}}(T^T \\mathbf{p}, \\tilde{Y})", "(2.5)")
    add_paragraph(
        doc,
        "In recommender systems, Gao et al. (2022) proposed T-CE for denoising implicit feedback, estimating transition probabilities based on user "
        "interaction frequencies. Wang et al. (2021) explored sample reweighting for explicit ratings. However, existing noise transition frameworks "
        "suffer from severe theoretical and practical deficiencies when applied to web graphs:"
    )
    add_bullet(doc, "Spectral Instability: When the estimated transition matrix $T$ is ill-conditioned or nearly singular (which occurs frequently in sparse recommendation data), naive matrix inversion $T^{-1}$ in backward correction causes unbounded gradient explosion, destabilizing neural network optimization.", bold_prefix="Issue 1: ")
    add_bullet(doc, "Stationarity Assumption: Standard transition models assume label noise is independently and identically distributed (i.i.d.) and spatially uniform across all users. They collapse under coordinated, non-stationary review-bombing bursts where noise is concentrated within specific temporal intervals.", bold_prefix="Issue 2: ")
    add_bullet(doc, "Neglect of Popularity Bias: Loss-correction models operate purely on classification risk without considering catalog exposure distributions, failing to prevent popularity bias from exacerbating data starvation.", bold_prefix="Issue 3: ")

    # 2.4 Popularity Bias & Long-Tail Calibration
    add_section_heading(doc, "2.4 Catalog Popularity Skew and Long-Tail Calibration")
    add_paragraph(
        doc,
        "Popularity bias represents an omnipresent structural distortion in recommender systems. In typical web catalogs, item consumption follows a power-law "
        "or Pareto distribution (Abdollahpouri et al., 2019; Anderson, 2006). A small fraction of 'head' items captures the overwhelming majority of interactions, "
        "while thousands of 'tail' items receive sparse engagement. Recommender systems trained on raw interaction graphs systematically over-recommend popular "
        "items and severely under-recommend niche products, a phenomenon driven by exposure bias, observational selection bias, and the rich-get-richer feedback loop."
    )
    add_paragraph(
        doc,
        "To mitigate popularity bias, several debiasing paradigms have been developed:"
    )
    add_bullet(doc, "Inverse Propensity Scoring (IPS): Weights training instances by the inverse probability of item exposure, $P(O_{ui} = 1)^{-1}$. While statistically unbiased, IPS suffers from extreme variance when exposure propensities for tail items are near zero (Schnabel et al., 2016).", bold_prefix="1. Propensity Weighting: ")
    add_bullet(doc, "Causal Graph Interventions: Uses do-calculus and counterfactual reasoning to disentangle user interest from item popularity shortcuts (Zheng et al., 2021; Zhang et al., 2021).", bold_prefix="2. Causal Disentanglement: ")
    add_bullet(doc, "Calibrated Recommendation: Pioneered by Steck (2018), calibration aims to ensure that the distribution of item categories or popularity tiers in a user's top-K recommendation list faithfully reflects the proportion of those tiers in their historical interactions. Abdollahpouri et al. (2021) formalized popularity calibration using KL divergence and ranking penalties.", bold_prefix="3. Popularity Calibration: ")

    # 2.5 Soft Blueprint Merging (SUPER/SUP)
    add_section_heading(doc, "2.5 Soft Blueprint Merging (SUPER/SUP) and Vulnerabilities")
    add_paragraph(
        doc,
        "A prominent breakthrough in popularity calibration is Soft Blueprint Merging for Calibrated Long-Tail Recommendation (SUPER / SUP), proposed by "
        "Yang et al. (2022). Standalone SUPER directly tackles catalog imbalance through an elegant divide-and-conquer strategy:"
    )
    add_numbered(doc, "Step 1:", "The catalog $\\mathcal{I}$ is partitioned at a Pareto cut-off (e.g., $\\rho = 0.20$) into popular Head items $\\mathcal{I}_{\\mathrm{head}}$ and long-tail Tail items $\\mathcal{I}_{\\mathrm{tail}}$.", bold_prefix="Catalog Partitioning: ")
    add_numbered(doc, "Step 2:", "Two specialized sub-models are trained independently: a Head Model $M_{\\mathrm{pop}}$ trained exclusively on head interactions, and a Tail Model $M_{\\mathrm{tail}}$ trained on tail interactions.", bold_prefix="Dual Model Training: ")
    add_numbered(doc, "Step 3:", "For each user $u$, a personalized tail preference inclination $\\alpha_u \\in [0, 1]$ is computed from their historical interaction blueprint $\\mathcal{B}_u$:", bold_prefix="Blueprint Inclination: ")
    add_equation(doc, "\\alpha_u = \\frac{|\\mathcal{B}_u \\cap \\mathcal{I}_{\\mathrm{tail}}|}{|\\mathcal{B}_u|}", "(2.6)")
    add_numbered(doc, "Step 4:", "During inference, candidate recommendation lists from $M_{\\mathrm{pop}}$ and $M_{\\mathrm{tail}}$ are soft-merged by dynamically interleaving items to satisfy each user's personalized tail quota $\\alpha_u \\cdot K$.", bold_prefix="Calibrated Merging: ")
    add_paragraph(
        doc,
        "The Critical Vulnerability of Standalone SUP: Despite its mathematical elegance, standalone SUPER is completely defenceless against adversarial data injection. "
        "SUPER treats the historical user blueprint $\\mathcal{B}_u$ as an immaculate ground-truth oracle. When adversarial bot accounts inject fake ratings on bandwagon tail targets, "
        "SUPER incorporates these fake interactions into the blueprints $\\mathcal{B}_u$. This corrupts $\\alpha_u$, inflates the perceived tail inclination, and forces the merger engine "
        "to fill the user's top-K recommendations with the attacker's spam targets! Thus, standalone SUPER actively worsens the damage of shilling attacks."
    )

    # 2.6 Graph Contrastive Learning
    add_section_heading(doc, "2.6 Graph Contrastive Learning (GCL: SGL, SimGCL)")
    add_paragraph(
        doc,
        "Self-supervised Graph Contrastive Learning (GCL) has recently achieved remarkable success in recommendation by learning robust representations "
        "without relying solely on sparse supervision labels. GCL maximizes mutual information between augmented views of the same node while minimizing "
        "mutual information across distinct nodes using the InfoNCE contrastive loss (van den Oord et al., 2018)."
    )
    add_paragraph(
        doc,
        "Wu et al. (2021) introduced Self-supervised Graph Learning (SGL), generating contrastive views via structural graph augmentations: Node Dropout "
        "(randomly removing vertices), Edge Dropout (randomly masking edges with probability $p_{\\mathrm{drop}}$), and Random Walk (sub-graph sampling). While effective, "
        "SGL incurs substantial computational overhead due to repeated adjacency matrix reconstruction."
    )
    add_paragraph(
        doc,
        "To overcome this overhead, Yu et al. (2022) proposed Simple Graph Contrastive Learning (SimGCL). SimGCL discards expensive structural augmentations "
        "entirely, showing that contrastive learning's primary benefit stems from representation uniformity across the hypersphere. SimGCL generates perturbed views "
        "by adding uniform random noise directly into latent embedding spaces:"
    )
    add_equation(doc, "\\mathbf{z}_i' = \\mathbf{e}_i + \\epsilon \\cdot \\mathrm{sgn}(\\mathbf{e}_i) \\odot \\mathbf{\\Delta}', \\quad \\mathbf{\\Delta}' \\sim \\mathcal{U}(0, 1)", "(2.7)")
    add_paragraph(
        doc,
        "However, existing GCL methods optimize an unweighted InfoNCE loss over all mini-batch interactions. When the graph contains adversarial edges, "
        "poisoned interactions are treated as authentic positive pairs, pulling corrupted representations closer together and distorting embedding geometry."
    )

    # 2.7 Trustworthy Graph Learning & Trust-GRS Positioning
    add_section_heading(doc, "2.7 Trustworthy Graph Learning and Positioning vs. Trust-GRS")
    add_paragraph(
        doc,
        "The intersection of graph neural networks and adversarial robustness has fostered the emerging sub-field of Trustworthy Graph Recommendation. "
        "Most notably, Trust-GRS (2025) proposed a contrastive framework designed to withstand shilling attacks by identifying structural anomalies in graph embeddings."
    )
    add_paragraph(
        doc,
        "Our proposed RRFN-LLM-SUPER framework fundamentally distinguishes itself from and advances beyond Trust-GRS in three decisive aspects:"
    )
    add_numbered(doc, "1. Joint Denoising and Pareto Debiasing:", "Trust-GRS focuses exclusively on graph embedding robustness, completely ignoring catalog popularity bias. In contrast, RRFN-LLM-SUPER uniquely couples robust noise-filtered representation learning with purified Pareto partition calibration.", bold_prefix="Advancement 1: ")
    add_numbered(doc, "2. Tri-View Multi-Modal Auditing (MIRF):", "While Trust-GRS relies solely on graph topological views, our MIRF ensemble synthesizes statistical rating transition matrices ($R_{\\mathrm{RRFN}}$), semantic natural language review audits via LLMs ($R_{\\mathrm{LLM}}$), and high-performance 2D BLAS temporal-polarity burst detection ($R_{\\mathrm{bomb}}$).", bold_prefix="Advancement 2: ")
    add_numbered(doc, "3. Mathematical Stability and Risk Consistency Proofs:", "Trust-GRS lacks theoretical operator norm bounds. We provide rigorous mathematical proofs establishing non-singular Tikhonov operator bounds ($\\|T_\\lambda^\\dagger\\|_2 \\le \\frac{1}{2\\sqrt{\\lambda}}$) and asymptotic surrogate excess risk consistency.", bold_prefix="Advancement 3: ")

    # 2.8 Large Language Models in Recommendation
    add_section_heading(doc, "2.8 Large Language Models (LLMs) in Recommendation Auditing")
    add_paragraph(
        doc,
        "The advent of powerful Large Language Models (LLMs)—such as OpenAI GPT-4, Anthropic Claude, and Meta LLaMA—has opened new frontiers in recommender systems "
        "(Fan et al., 2023; Wu et al., 2023). While early literature explored LLMs as generative recommendation rankers, real-world deployment is severely constrained "
        "by prohibitive inference latency (hundreds of milliseconds per query), API financial costs, and concurrency bottlenecks."
    )
    add_paragraph(
        doc,
        "In this work, rather than employing LLMs for continuous candidate ranking, we strategically deploy LLMs as high-precision, asynchronous semantic audit oracles. "
        "When an interaction edge contains natural language review text S_ui, an LLM evaluates the semantic consistency between the review sentiment and the numeric rating r_ui, "
        "identifying automated bot templates, synthetic astroturfing phrasing, and rating-text polarity contradictions. To eliminate latency bottlenecks during model training, "
        "all LLM audit outputs are indexed in an ACID-compliant, thread-safe SQLite database configured with Write-Ahead Logging (WAL) for O(1) cache retrieval."
    )

    # 2.9 Research Gaps
    add_section_heading(doc, "2.9 Critical Research Gaps in Prior Literature")
    add_paragraph(
        doc,
        "In summary, our exhaustive literature survey reveals four fundamental research gaps in existing recommendation literature:"
    )
    add_bullet(doc, "Research Gap 1 (Isolated Vulnerability Treatment): No existing framework simultaneously solves adversarial shilling attacks, temporal review-bombing bursts, and catalog popularity bias. Existing methods collapse when attacks exploit popularity bias.", bold_prefix="Gap 1: ")
    add_bullet(doc, "Research Gap 2 (Lack of Non-Singular Mathematical Guarantees): Prior noise transition models rely on unregularized matrix inversions that explode numerically under ill-conditioned empirical rating matrices.", bold_prefix="Gap 2: ")
    add_bullet(doc, "Research Gap 3 (Computational Bottlenecks in Multi-Modal Filtering): Prior burst detection and LLM auditing methods suffer from extreme runtime overhead (>10 seconds per epoch), making them unusable for real-world graph training.", bold_prefix="Gap 3: ")
    add_bullet(doc, "Research Gap 4 (Flawed Evaluation Metrics): Existing long-tail metrics (raw LTC) treat pushed attack targets as successful tail discovery, masking severe spam exposure and misreporting attack success as algorithmic fairness.", bold_prefix="Gap 4: ")

    # 2.10 Literature Positioning Matrix (Table 2.1)
    add_section_heading(doc, "2.10 Comprehensive Literature Positioning Matrix")
    add_paragraph(
        doc,
        "To explicitly position RRFN-LLM-SUPER within the broader research landscape, Table 2.1 presents a comprehensive comparative feature matrix "
        "evaluating eight representative paradigms across eight critical technical dimensions."
    )

    headers = [
        "Paradigm / Model", "Adversarial Robustness", "Popularity Debiasing", "Graph Learning",
        "Multi-Modal Auditing", "Risk Consistency", "Operator Stability", "Vectorized Burst"
    ]
    data = [
        ["Standard Matrix Factorization (Koren, 2009)", "No (Vulnerable)", "No (Heavily Biased)", "No (Pairwise)", "No", "No", "N/A", "No"],
        ["Neural Collaborative Filtering (He, 2017)", "No (Vulnerable)", "No (Heavily Biased)", "No (MLP)", "No", "No", "N/A", "No"],
        ["LightGCN (He, 2020)", "No (Amplifies Noise)", "No (Biased)", "Yes (High-Order)", "No", "No", "N/A", "No"],
        ["T-CE Implicit Denoising (Gao, 2022)", "Partial (Implicit Only)", "No", "No", "No", "Heuristic", "No (Unbounded)", "No"],
        ["Standalone SUPER / SUP (Yang, 2022)", "No (Severe Inflation)", "Yes (Calibrated Quotas)", "No", "No", "No", "N/A", "No"],
        ["SimGCL / SGL (Yu, 2022; Wu, 2021)", "No (Unweighted InfoNCE)", "No", "Yes (Contrastive)", "No", "No", "N/A", "No"],
        ["Trust-GRS (2025)", "Yes (Graph Contrastive)", "No (Ignores Tail)", "Yes (GNN Defense)", "No (Topology Only)", "No Proof", "No", "No"],
        ["RRFN-LLM-SUPER (Ours)", "Yes (Full Defense)", "Yes (Purified SUPER)", "Yes (Risk-InfoNCE)", "Yes (MIRF: LLM+BLAS)", "Proven Theorem 2", "Proven Theorem 1", "Yes (<5ms BLAS)"]
    ]
    col_widths = [3.8, 1.8, 1.8, 1.7, 1.8, 1.7, 1.6, 1.4]
    add_table_custom(
        doc, headers, data,
        caption_text="Comprehensive Literature Positioning Matrix Comparing Existing Paradigms Against RRFN-LLM-SUPER Across 8 Technical Dimensions.",
        table_num_str="2.1",
        col_widths=col_widths
    )

    doc.add_page_break()
