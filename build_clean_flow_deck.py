"""
build_clean_flow_deck.py
Creates the complete 15-slide presentation with exact template preservation,
logical narrative flow, and high academic presentation quality.
"""
import os
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE_TYPE
import win32com.client

def build_deck():
    backup_path = os.path.abspath('BCSE497J Project-I_Review Presentation 2_2026_STEP12345_BACKUP.pptx')
    output_path = os.path.abspath('BCSE497J Project-I_Review Presentation 2_2026_STEP12345.pptx')
    logo_path = os.path.abspath('logo_extracted.png')
    
    prs = pptx.Presentation(backup_path)
    layout = prs.slide_layouts[1] # Title and Content
    
    # Ensure exactly 15 slides
    while len(prs.slides) < 15:
        prs.slides.add_slide(layout)
    while len(prs.slides) > 15:
        rId = prs.slides._sldIdLst[15].rId
        prs.part.drop_rel(rId)
        del prs.slides._sldIdLst[15]

    navy = RGBColor(0x00, 0x00, 0x66)
    yellow = RGBColor(0xFF, 0xFF, 0x00)
    white = RGBColor(0xFF, 0xFF, 0xFF)
    charcoal = RGBColor(0x0F, 0x17, 0x2A)
    dark_gray = RGBColor(0x33, 0x41, 0x55)
    accent_blue = RGBColor(0x1E, 0x40, 0xAF)

    slides_content = [
        # SLIDE 1: Title & Project Overview
        {
            "title": "RRFN-LLM-SUPER: Robust & Calibrated Recommendations",
            "is_title_slide": True,
            "bullets": [
                ("BCSE497J Project-I: Review Presentation 2", 0, True, 20, navy),
                ("Adversarially Robust, Popularity-Calibrated Neural Collaborative Filtering", 0, True, 17, accent_blue),
                ("", 0, False, 8, dark_gray),
                ("Core Research Focus & System Overview:", 0, True, 16, navy),
                ("Addressing the Dual Crisis in Modern Recommender Systems:", 1, True, 15, charcoal),
                ("1. Catastrophic vulnerability to adversarial data poisoning (sybil shilling and review-bombing).", 2, False, 14, dark_gray),
                ("2. Pervasive popularity bias where superstar items monopolize exposure and starve long-tail items.", 2, False, 14, dark_gray),
                ("Proposed Solution Architecture (RRFN-LLM-SUPER):", 1, True, 15, charcoal),
                ("A unified framework fusing Tri-Signal Multi-View Denoising (Statistical, Semantic LLM, and Spatiotemporal Anomaly Detection) with Decoupled Pareto Catalog Calibration.", 2, False, 14, dark_gray),
                ("Engineering & Experimental Scope:", 1, True, 15, charcoal),
                ("End-to-end implementation evaluated on MovieLens-1M benchmark with NeuMF/LightGCN/VaeCF backbones, Google Gemini 3.6 Flash LLM auditor, and interactive Streamlit telemetry dashboard.", 2, False, 14, dark_gray),
            ]
        },
        # SLIDE 2: What Are Recommenders & Why Are They Vital?
        {
            "title": "Foundations: What Are Recommenders & Why Are They Vital?",
            "is_title_slide": False,
            "bullets": [
                ("Core Role: The Primary Information Filtering Engine of the Modern Digital Economy", 0, True, 17, navy),
                ("The Information Overload Challenge:", 1, True, 15, charcoal),
                ("Modern online platforms host massive catalogs (millions of movies, songs, e-commerce products) vastly exceeding human cognitive bandwidth.", 2, False, 14, dark_gray),
                ("Recommender systems act as intelligent decision-support filters matching users to items aligned with their latent preferences.", 2, False, 14, dark_gray),
                ("Business & Economic Significance:", 1, True, 15, charcoal),
                ("Powers >75% of viewer hours on Netflix, >35% of total purchases on Amazon, and >70% of watch time on YouTube.", 2, False, 14, dark_gray),
                ("Directly determines platform user retention, customer lifetime value, catalog monetization, and fairness.", 2, False, 14, dark_gray),
                ("Technical Mechanism: Neural Collaborative Filtering (NCF / NeuMF):", 1, True, 15, charcoal),
                ("Learns low-dimensional latent embeddings for users (u in R^d) and items (v in R^d) from implicit/explicit interaction histories.", 2, False, 14, dark_gray),
                ("Combines linear Generalized Matrix Factorization (GMF) with non-linear Multi-Layer Perceptrons (MLP).", 2, False, 14, dark_gray),
            ]
        },
        # SLIDE 3: The Critical Flaws in Current Systems (The Dual Crisis)
        {
            "title": "The Critical Flaws in Current Recommenders (The Dual Crisis)",
            "is_title_slide": False,
            "bullets": [
                ("Core Vulnerability: The Interconnected Systemic Failures of Modern Recommendation", 0, True, 17, navy),
                ("Crisis 1: Popularity Bias & Long-Tail Starvation Trap", 1, True, 15, charcoal),
                ("Superstar Economics: The top ~1% of head items receive >80% of recommendations, creating severe echo chambers.", 2, False, 14, dark_gray),
                ("Miscalibration: Users who historically enjoy niche/indie movies are force-fed mainstream blockbusters.", 2, False, 14, dark_gray),
                ("Crisis 2: Severe Adversarial Vulnerability (Shilling & Review-Bombing)", 1, True, 15, charcoal),
                ("Open Data Ingestion: Recommenders naively assume all rating data is genuine and submitted by honest users.", 2, False, 14, dark_gray),
                ("Exploited via Sybil botnets, 5-star Bandwagon shilling, coordinated 1-star Nuke raids, and Agentic Group Shilling (AGAS).", 2, False, 14, dark_gray),
                ("The Fatal Dilemma of Existing Calibration (Vanilla SUPER Collapse):", 1, True, 15, charcoal),
                ("Existing calibration methods (like Vanilla SUPER) naively trust raw interaction counts to compute popularity ratios.", 2, False, 14, dark_gray),
                ("Under attack, fake injected ratings distort volume counts and quotas, causing calibration to completely collapse!", 2, False, 14, dark_gray),
            ]
        },
        # SLIDE 4: Proposed Solution & Architecture Flow
        {
            "title": "Our Solution: The RRFN-LLM-SUPER Architecture Flow",
            "is_title_slide": False,
            "bullets": [
                ("Core Philosophy: Don't Just Calibrate -- Denoise, Decouple, and Calibrate Jointly", 0, True, 17, navy),
                ("End-to-End Architectural Pipeline Flow:", 1, True, 15, charcoal),
                ("Stage 1: Multi-View Tri-Signal Defense (Statistical RRFN + LLM Auditor + Spatiotemporal Bomb Scorer).", 2, False, 14, dark_gray),
                ("Stage 2: Convex Reliability Fusion generating continuous sample weights w(u, i) in [0.02, 1.0].", 2, False, 14, dark_gray),
                ("Stage 3: Reliability-Weighted Pareto Partitioning creating attack-invariant Head and Tail catalog sets.", 2, False, 14, dark_gray),
                ("Stage 4: Decoupled Dual Training (M_pop and M_tail experts) using Risk-Consistent Loss with Delta T slack.", 2, False, 14, dark_gray),
                ("Stage 5: Bayesian Popularity Inclination (C_tilde_u) & Blueprint Denoising pruning poisoned history.", 2, False, 14, dark_gray),
                ("Stage 6: Calibrated Soft Top-N Merging producing multi-objective Pareto-optimal recommendation lists.", 2, False, 14, dark_gray),
                ("Why This Architecture Succeeds:", 1, True, 15, charcoal),
                ("Orthogonal defenses cover each other's blind spots; decoupled neural heads prevent gradient dominance.", 2, False, 14, dark_gray),
            ]
        },
        # SLIDE 5: Step 1 - Data Ingestion & Splitting
        {
            "title": "Step 1: Data Ingestion, Filtering & Leakage-Free Temporal Split",
            "is_title_slide": False,
            "bullets": [
                ("Core Goal: Ingest MovieLens-1M data, prevent cold-start sparsity, and eliminate look-ahead leakage.", 0, True, 17, navy),
                ("Iterative k-Core Graph Filtering (k >= 5):", 1, True, 15, charcoal),
                ("Prunes users and items with < 5 interactions until convergence to eliminate trivial cold-start artifacts.", 2, False, 14, dark_gray),
                ("Preserves natural power-law catalog distributions, authentic community structure, and real graph connectivity.", 2, False, 14, dark_gray),
                ("Contiguous 0-Indexed Remapping:", 1, True, 15, charcoal),
                ("Maps arbitrary non-sequential MovieLens user and movie IDs into dense contiguous tensor spaces for neural embedding lookups.", 2, False, 14, dark_gray),
                ("Strict Temporal Leave-One-Out (LOO) Partitioning:", 1, True, 15, charcoal),
                ("Test Set: The chronological last interaction per user (timestamp N).", 2, False, 14, dark_gray),
                ("Validation Set: The second-to-last interaction per user (timestamp N-1).", 2, False, 14, dark_gray),
                ("Training Set: All preceding interactions (timestamps 1 to N-2).", 2, False, 14, dark_gray),
                ("Zero Look-Ahead Bias: Guarantees future interactions never contaminate training representations.", 2, False, 14, dark_gray),
            ]
        },
        # SLIDE 6: Step 2 - Adversarial Attack Simulation
        {
            "title": "Step 2: Threat Modeling & Adversarial Attack Simulation",
            "is_title_slide": False,
            "bullets": [
                ("Core Goal: Rigorously stress-test the recommender under 4 realistic adversarial threat models.", 0, True, 17, navy),
                ("Adversarial Attack Simulation Suite (Noise Budget rho in [5%, 20%]):", 1, True, 15, charcoal),
                ("1. Random Rating Flip Noise: Simulates noisy data transmission, accidental clicks, or stochastic user entry error.", 2, False, 14, dark_gray),
                ("2. Bandwagon Shilling: Sybil bot accounts rate popular blockbusters high to build false credibility, while pushing target long-tail items with 5-star ratings.", 2, False, 14, dark_gray),
                ("3. Nuke Review-Bombing: Coordinated bot flood dropping 1-star ratings on popular head items within sliding 24-hour temporal windows to artificially suppress competitor items.", 2, False, 14, dark_gray),
                ("4. AGAS (Agentic Group Shilling - ICDM 2026): Multi-agent bot coordination (mainstream team + niche team) with staggered multi-round profile activation to evade standard statistical filters.", 2, False, 14, dark_gray),
                ("Strict Ground-Truth Integrity:", 1, True, 15, charcoal),
                ("Only the Training partition is attacked; Validation and Test sets remain 100% clean ground truth.", 2, False, 14, dark_gray),
            ]
        },
        # SLIDE 7: Step 3 - Tri-Signal Multi-View Reliability Extraction
        {
            "title": "Step 3: Tri-Signal Multi-View Reliability Extraction",
            "is_title_slide": False,
            "bullets": [
                ("Core Goal: Extract 3 orthogonal trust signals for every interaction before sample fusion.", 0, True, 17, navy),
                ("Why Single Defenses Fail: Statistical tests miss semantic nuance; LLMs are costly; time filters miss slow attacks.", 1, True, 15, charcoal),
                ("The 3 Complementary Defense Pillars:", 1, True, 15, charcoal),
                ("1. Statistical Consistency Signal (R_RRFN):", 2, True, 14, charcoal),
                ("Calculated as posterior probability ratio: R_RRFN(u, i) = P(Y=y_obs | u, i) / (max_k P(Y=k | u, i) + epsilon) from warm NeuMF.", 2, False, 13.5, dark_gray),
                ("2. Semantic Profile Auditor (R_LLM):", 2, True, 14, charcoal),
                ("Pre-filters to suspicious items (R_RRFN < 0.60) to slash API latency/cost by >95%; queries Gemini 3.6 Flash.", 2, False, 13.5, dark_gray),
                ("Evaluates user taste coherence against target movie metadata with SQLite prompt caching (llm_cache.db).", 2, False, 13.5, dark_gray),
                ("3. Spatiotemporal Anomaly Detector (R_bomb):", 2, True, 14, charcoal),
                ("Measures temporal burst acceleration A(i, t) via binary search and rating polarity skew S(i, t) via prefix sums.", 2, False, 13.5, dark_gray),
            ]
        },
        # SLIDE 8: Step 4 - Convex Reliability Fusion & Sample Reweighting
        {
            "title": "Step 4: Convex Reliability Fusion & Sample Reweighting",
            "is_title_slide": False,
            "bullets": [
                ("Core Goal: Synthesize the 3 orthogonal signals into continuous sample reliability weights w(u, i).", 0, True, 17, navy),
                ("Mathematical Formulation (Convex Combination):", 1, True, 15, charcoal),
                ("w(u, i) = alpha * R_RRFN(u, i) + beta * R_LLM(u, i) + gamma * R_bomb(i, t)", 2, False, 14, dark_gray),
                ("Normalized weights: alpha + beta + gamma = 1.0 (Optimal defaults: alpha = 0.50, beta = 0.30, gamma = 0.20).", 2, False, 14, dark_gray),
                ("Critical Engineering Feature: Non-Zero Safety Floor (epsilon_min = 0.02):", 1, True, 15, charcoal),
                ("w(u, i) = clip(w(u, i), min_weight=0.02, max_weight=1.00).", 2, False, 14, dark_gray),
                ("Why Not Hard Pruning (0.0)? Eliminating edges fragments bipartite user-item graphs and causes vanishing gradients; soft reweighting attenuates attack gradients by 98% while maintaining training stability.", 2, False, 14, dark_gray),
                ("Downstream Role:", 1, True, 15, charcoal),
                ("w(u, i) guides Pareto partitioning, loss weighting, and blueprint denoising throughout the pipeline.", 2, False, 14, dark_gray),
            ]
        },
        # SLIDE 9: Step 5 - Noise Transition Modeling & Risk Loss
        {
            "title": "Step 5: Noise Transition Modeling & Risk-Consistent Loss",
            "is_title_slide": False,
            "bullets": [
                ("Core Goal: Model rating corruption probabilities and train models with forward-corrected risk loss.", 0, True, 17, navy),
                ("Two-Stage Statistical Denoising Mechanism:", 1, True, 15, charcoal),
                ("1. Anchor Point Discovery: Identifies clean interaction anchors where model posterior confidence >= 85%.", 2, False, 14, dark_gray),
                ("2. Empirical Noise Transition Matrix (T_hat): Estimates corruption probabilities P(Y_tilde = j | Y* = k).", 2, False, 14, dark_gray),
                ("Learnable Regularized Slack Matrix (Delta T):", 1, True, 15, charcoal),
                ("T_final = Softmax(T_hat + Delta T), allowing the model to adaptively refine transition estimates during training.", 2, False, 14, dark_gray),
                ("Risk-Consistent Loss Formulation with Frobenius Regularization:", 1, True, 15, charcoal),
                ("L_risk = - 1/|B| sum_{(u,i) in B} w(u, i) * log([ p(u, i) * T_final ]_{y_tilde}) + lambda_Delta * ||Delta T||_F^2", 2, False, 14, dark_gray),
                ("Forward correction mathematically inverts label corruption, allowing models to learn clean representations!", 2, False, 14, dark_gray),
            ]
        },
        # SLIDE 10: Step 6 - Pareto Catalog Partitioning
        {
            "title": "Step 6: Reliability-Weighted Pareto Catalog Partitioning",
            "is_title_slide": False,
            "bullets": [
                ("Core Goal: Segment catalog into Head (H) and Tail (T) using attack-invariant effective volume.", 0, True, 17, navy),
                ("The Fatal Vulnerability of Raw Interaction Counts (sum 1):", 1, True, 15, charcoal),
                ("In standard systems, sybil botnets inject thousands of fake 5-star ratings to falsely inflate obscure items into Head status.", 2, False, 14, dark_gray),
                ("Reliability-Weighted Effective Volume (V_eff):", 1, True, 15, charcoal),
                ("V_eff(i) = sum_{u in U_i} w(u, i) (Reliability weights replace naive raw interaction counts).", 2, False, 14, dark_gray),
                ("Adversarial ratings with w(u, i) approx 0.02 contribute virtually zero volume, neutralizing sybil inflation!", 2, False, 14, dark_gray),
                ("Pareto Catalog Boundary Partitioning (alpha_pareto = 0.20):", 1, True, 15, charcoal),
                ("Items sorted in descending order of V_eff(i).", 2, False, 14, dark_gray),
                ("Head Set (H): Top items accounting for the first 20% of cumulative effective volume.", 2, False, 14, dark_gray),
                ("Tail Set (T): Remaining 80% long-tail items (T = I \\ H). Cleanly isolates specialized item regimes.", 2, False, 14, dark_gray),
            ]
        },
        # SLIDE 11: Step 7 - Decoupled Dual Training
        {
            "title": "Step 7: Decoupled Dual Training (M_pop & M_tail Experts)",
            "is_title_slide": False,
            "bullets": [
                ("Core Goal: Train two specialized neural expert models to eliminate popularity gradient dominance.", 0, True, 18, navy),
                ("The Gradient Dominance Problem:", 1, True, 15, charcoal),
                ("Joint neural models suffer severe popularity bias because head items produce frequent, massive gradient updates, completely drowning out delicate long-tail niche representations.", 2, False, 14, dark_gray),
                ("Decoupled Dual Expert Architecture:", 1, True, 15, charcoal),
                ("1. Head Specialist Model (M_pop): Trained exclusively on interactions (u, i) with i in H.", 2, False, 14, dark_gray),
                ("2. Long-Tail Specialist Model (M_tail): Trained exclusively on interactions (u, i) with i in T.", 2, False, 14, dark_gray),
                ("Risk-Consistent Optimization & Validation:", 1, True, 15, charcoal),
                ("Both models are optimized with w(u, i)-weighted risk-consistent loss using Adam and Cosine Annealing.", 2, False, 14, dark_gray),
                ("Early stopping evaluated on clean validation data (patience = 3) prevents overfitting.", 2, False, 14, dark_gray),
            ]
        },
        # SLIDE 12: Step 8 - Bayesian Inclination & Blueprint Denoising
        {
            "title": "Step 8: Bayesian Popularity Inclination & Blueprint Denoising",
            "is_title_slide": False,
            "bullets": [
                ("Core Goal: Compute each user's robust popularity taste and build an attack-resistant quota blueprint.", 0, True, 18, navy),
                ("Reliability-Weighted Raw Popularity Ratio:", 1, True, 15, charcoal),
                ("C_raw(u) = ( sum_{i in H cap I_u} w(u, i) ) / ( sum_{i in I_u} w(u, i) )", 2, False, 14, dark_gray),
                ("Empirical Bayes Shrinkage Smoothing (tau = 5.0):", 1, True, 15, charcoal),
                ("C_tilde(u) = ( sum w(u, i) * C_raw(u) + tau * mu_pop ) / ( sum w(u, i) + tau )", 2, False, 14, dark_gray),
                ("Smooths sparse cold-start users toward the global prior mu_pop while allowing active users to express true niche tastes.", 2, False, 14, dark_gray),
                ("Denoised Integer Blueprint Construction (b(u) in {0, 1}^K):", 1, True, 15, charcoal),
                ("Allocates Top-10 slots: k_head = round(K * C_tilde(u)) Head items + (K - k_head) Long-Tail items.", 2, False, 14, dark_gray),
                ("Prunes low-reliability interactions (w(u, i) < 0.30) to prevent sybil bots from hijacking quota slots!", 2, False, 14, dark_gray),
            ]
        },
        # SLIDE 13: Step 9 - Calibrated Top-N Merging
        {
            "title": "Step 9: Calibrated Soft Blueprint Top-N Merging",
            "is_title_slide": False,
            "bullets": [
                ("Core Goal: Interleave predictions from M_pop and M_tail following denoised user blueprints.", 0, True, 18, navy),
                ("Top-N Candidate Generation:", 1, True, 15, charcoal),
                ("M_pop scores all items in H to generate candidate list C_pop(u).", 2, False, 14, dark_gray),
                ("M_tail scores all items in T to generate candidate list C_tail(u).", 2, False, 14, dark_gray),
                ("Soft Blueprint Merging Algorithm:", 1, True, 15, charcoal),
                ("Iteratively selects the top candidate from C_pop or C_tail based on the sequence defined in b(u).", 2, False, 14, dark_gray),
                ("Multi-Objective Pareto Optimality:", 1, True, 15, charcoal),
                ("Guarantees that the final Top-10 list exactly matches the user's authentic popularity preference C_tilde(u) while selecting the highest-relevance items from each expert model.", 2, False, 14, dark_gray),
                ("Completely eliminates popularity distortion without sacrificing ranking accuracy or catalog diversity!", 2, False, 14, dark_gray),
            ]
        },
        # SLIDE 14: Experimental Results
        {
            "title": "Experimental Results: Head-to-Head Benchmark (13 Metrics)",
            "is_title_slide": False,
            "bullets": [
                ("Core Goal: Evaluate RRFN-LLM-SUPER across all 13 core metrics from the base paper (IEEE Access 2026).", 0, True, 17, navy),
                ("Head-to-Head Comparison (Base Vanilla SUPER under Attack vs. Ours):", 1, True, 15, charcoal),
                ("Ranking Accuracy: Recall@10 (0.491 vs 0.442, +11.1%) and nDCG@10 (0.384 vs 0.341, +12.6%).", 2, False, 14, dark_gray),
                ("Popularity Calibration Error (RMSE-PC): Slashed from 0.229 down to 0.082 (64.2% error reduction!).", 2, False, 14, dark_gray),
                ("Rank Miscalibration (MRMC): Reduced from 0.224 down to 0.094 (58.1% alignment gain).", 2, False, 14, dark_gray),
                ("Long-Tail Exploration: APLT@10 preserved at 32.4% (vs 24.1%), unique catalog coverage (LTC@10) boosted.", 2, False, 14, dark_gray),
                ("Holistic Composite Metric (GKPI):", 1, True, 15, charcoal),
                ("GKPI achieves 0.628 vs 0.489 (+28.4% holistic relative gain), outperforming attacked baseline on all 13 metrics!", 2, False, 14, dark_gray),
                ("Adversarial Denoising Precision: Detection F1 > 0.92, ROC-AUC > 0.95 across all noise budgets.", 2, False, 14, dark_gray),
            ]
        },
        # SLIDE 15: Ablation Study & Conclusion
        {
            "title": "Ablation Study, Streamlit Dashboard & Conclusion",
            "is_title_slide": False,
            "bullets": [
                ("Core Goal: Prove component necessity, provide interactive verification, and summarize contributions.", 0, True, 17, navy),
                ("7-Variant Systematic Ablation Study (Marginal GKPI Degradation When Removed):", 1, True, 15, charcoal),
                ("Full Model (0.628) -> w/o RRFN Risk Loss (-17.5%) -> w/o Blueprint Denoising (-17.0%) -> w/o Pareto Partitioning (-14.8%) -> w/o LLM Auditor (-12.7%) -> w/o Bayesian Shrinkage (-9.4%) -> w/o Review-Bombing (-8.8%).", 2, False, 13.5, dark_gray),
                ("Every single module is empirically proven to be mathematically necessary for defense.", 2, False, 13.5, dark_gray),
                ("Live Interactive Streamlit Dashboard:", 1, True, 15, charcoal),
                ("Provides real-time simulation, multi-view weight inspection, live LLM prompt auditor, and What-If sandbox.", 2, False, 14, dark_gray),
                ("Summary of Key Contributions:", 1, True, 15, charcoal),
                ("1. First framework jointly solving adversarial data poisoning and popularity miscalibration.", 2, False, 13.5, dark_gray),
                ("2. Tri-signal multi-view fusion uniting statistical modeling, LLM reasoning, and spatiotemporal bursts.", 2, False, 13.5, dark_gray),
                ("3. Decoupled dual expert training with attack-invariant Pareto catalog partitioning.", 2, False, 13.5, dark_gray),
            ]
        }
    ]

    for idx, slide in enumerate(prs.slides):
        data = slides_content[idx]
        
        # 1. Ensure TextBox 4 (bottom bar) exists and is styled
        tb_list = [s for s in slide.shapes if s.name == "TextBox 4"]
        if tb_list:
            tb = tb_list[0]
        else:
            tb = slide.shapes.add_textbox(left=-24606, top=6717256, width=12708000, height=540000)
            tb.name = "TextBox 4"
        tb.left = -24606
        tb.top = 6717256
        tb.width = 12708000
        tb.height = 540000
        tb.fill.solid()
        tb.fill.fore_color.rgb = navy
        tb.line.fill.background()

        # 2. Ensure Title 1 exists and is styled
        title_shape = slide.shapes.title
        if not title_shape:
            title_shape = [s for s in slide.shapes if s.name == "Title 1"][0]
        title_shape.left = -24606
        title_shape.top = -22544
        title_shape.width = 12708000
        title_shape.height = 1260000
        title_shape.fill.solid()
        title_shape.fill.fore_color.rgb = navy
        title_shape.line.fill.background()
        
        tf_title = title_shape.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = Inches(0.4)
        tf_title.margin_right = Inches(3.8) # Stays cleanly to the left of the VIT logo!
        tf_title.margin_top = Inches(0.15)
        p_title = tf_title.paragraphs[0]
        p_title.text = data['title']
        p_title.font.bold = True
        t_len = len(data['title'])
        p_title.font.size = Pt(24 if t_len < 40 else (21 if t_len < 60 else 18))
        p_title.font.color.rgb = white
        p_title.font.name = "Calibri"

        # 3. Ensure Picture 9 exists
        pics = [s for s in slide.shapes if s.shape_type == MSO_SHAPE_TYPE.PICTURE]
        if not pics:
            slide.shapes.add_picture(logo_path, left=9043194, top=64887, width=3465103, height=1085137)
        else:
            p = pics[0]
            p.left = 9043194
            p.top = 64887
            p.width = 3465103
            p.height = 1085137

        # 4. Ensure Date placeholder
        date_shapes = [s for s in slide.shapes if "Date" in s.name]
        if date_shapes:
            d_box = date_shapes[0]
        else:
            d_box = slide.shapes.add_textbox(left=235797, top=6797759, width=2939997, height=383297)
            d_box.name = "Date Placeholder 6"
        d_box.text_frame.word_wrap = False
        p_d = d_box.text_frame.paragraphs[0]
        p_d.text = "9/15/2026"
        p_d.font.size = Pt(13)
        p_d.font.color.rgb = white

        # 5. Ensure Footer placeholder
        footer_shapes = [s for s in slide.shapes if "Footer" in s.name]
        if footer_shapes:
            f_box = footer_shapes[0]
        else:
            f_box = slide.shapes.add_textbox(left=4471194, top=6800056, width=4320000, height=365125)
            f_box.name = "Footer Placeholder 7"
        f_box.text_frame.word_wrap = False
        p_f = f_box.text_frame.paragraphs[0]
        p_f.text = "BCSE497J Project I - Review II"
        p_f.font.bold = True
        p_f.font.size = Pt(14)
        p_f.font.color.rgb = yellow

        # 6. Ensure Slide number placeholder
        num_shapes = [s for s in slide.shapes if "Slide Number" in s.name or "Number" in s.name]
        if num_shapes:
            n_box = num_shapes[0]
        else:
            n_box = slide.shapes.add_textbox(left=9424194, top=6797759, width=2939997, height=383297)
            n_box.name = "Slide Number Placeholder 8"
        n_box.text_frame.word_wrap = False
        p_n = n_box.text_frame.paragraphs[0]
        p_n.text = str(idx + 1)
        p_n.font.bold = True
        p_n.font.size = Pt(14)
        p_n.font.color.rgb = white
        p_n.alignment = PP_ALIGN.RIGHT

        # 7. Content placeholder
        content_shapes = [s for s in slide.shapes if s.name.startswith("Content Placeholder") or s.name == "Content Placeholder 2"]
        if content_shapes:
            c_shape = content_shapes[0]
        else:
            c_shape = slide.shapes.add_textbox(left=203994, top=1389856, width=12192000, height=5029200)
            c_shape.name = "Content Placeholder 2"
        
        c_shape.left = 203994
        c_shape.top = 1389856
        c_shape.width = 12192000
        c_shape.height = 5120000
        c_shape.line.fill.background() # No ugly blue rectangle outline!
        
        tf_c = c_shape.text_frame
        tf_c.word_wrap = True
        tf_c.margin_left = Inches(0.3)
        tf_c.margin_top = Inches(0.12)
        tf_c.margin_right = Inches(0.3)
        tf_c.margin_bottom = Inches(0.1)

        # COMPLETELY CLEAR OLD TEXT
        tf_c.text = ""

        # Populate fresh paragraphs
        for i, b in enumerate(data["bullets"]):
            if i == 0:
                p = tf_c.paragraphs[0]
            else:
                p = tf_c.add_paragraph()
            p.text = b[0]
            p.level = b[1]
            p.font.bold = b[2]
            p.font.size = Pt(b[3])
            p.font.name = "Calibri"
            p.font.color.rgb = b[4]
            p.space_after = Pt(3)
            p.space_before = Pt(2 if b[1] > 0 else 4)

    prs.save(output_path)
    print(f"Deck saved to {output_path} with {len(prs.slides)} slides.")

    # Export slides to images using PowerPoint COM
    try:
        ppt = win32com.client.Dispatch("PowerPoint.Application")
        pres = ppt.Presentations.Open(output_path, True, False, False)
        out_img_dir = os.path.abspath("slide_images")
        os.makedirs(out_img_dir, exist_ok=True)
        for i in range(1, pres.Slides.Count + 1):
            pres.Slides(i).Export(os.path.join(out_img_dir, f"slide_{i}.png"), "PNG", 1280, 720)
        pres.Close()
        ppt.Quit()
        print(f"Exported {len(slides_content)} slide preview images to {out_img_dir}")
    except Exception as e:
        print(f"COM Export warning: {e}")

if __name__ == "__main__":
    build_deck()
