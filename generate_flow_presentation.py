"""
generate_flow_presentation.py
Rebuilds BCSE497J Project-I_Review Presentation 2_2026_STEP12345.pptx with a seamless
academic/engineering narrative flow:
1. Recommender foundations & significance
2. Critical flaws (Popularity bias & Adversarial vulnerability)
3. End-to-end architecture & multi-view solution flow
4. Step-by-step pipeline stages (Data, Attacks, Tri-Signal, Fusion, Loss, Pareto, Dual Models, Blueprints, Merging)
5. Comprehensive evaluation results & 7-variant ablation
6. Interactive dashboard & conclusion
"""
import sys
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE_TYPE

def build_presentation():
    # Load base presentation to inherit slide layouts, masters, and color palettes
    prs = pptx.Presentation('BCSE497J Project-I_Review Presentation 2_2026_STEP12345_BACKUP.pptx')
    
    # Remove existing slides
    xml_slides = prs.slides._sldIdLst
    for sld in list(xml_slides):
        xml_slides.remove(sld)

    navy_color = RGBColor(0x00, 0x00, 0x66)
    yellow_color = RGBColor(0xFF, 0xFF, 0x00)
    white_color = RGBColor(0xFF, 0xFF, 0xFF)
    dark_gray = RGBColor(0x33, 0x33, 0x33)

    slides_data = [
        # SLIDE 1
        {
            "title": "RRFN-LLM-SUPER: Robust & Calibrated Recommendations",
            "paragraphs": [
                ("Project-I Review Presentation II | Course Code: BCSE497J", 0, True, 22),
                ("Adversarially Robust, Popularity-Calibrated Neural Collaborative Filtering", 0, False, 18),
                ("", 0, False, 12),
                ("Core Research Motivation & Problem Statement:", 0, True, 18),
                ("Addressing the Dual Crisis in Modern Recommender Systems:", 1, True, 16),
                ("1. Catastrophic vulnerability to adversarial shilling and coordinated review-bombing attacks.", 2, False, 15),
                ("2. Pervasive popularity bias where superstar items starve the long-tail and distort user calibration.", 2, False, 15),
                ("Proposed Solution Framework:", 1, True, 16),
                ("A unified architecture uniting Tri-Signal Multi-View Reliability Denoising (Statistical, Semantic LLM, and Spatiotemporal Anomaly Detection) with Decoupled Pareto Catalog Calibration.", 2, False, 15),
                ("Project Scope:", 1, True, 16),
                ("End-to-end implementation tested on MovieLens-1M with PyTorch, Gemini 3.6 Flash, and Streamlit Telemetry.", 2, False, 15),
            ]
        },
        # SLIDE 2
        {
            "title": "Foundations: What Are Recommenders & Why Are They Vital?",
            "paragraphs": [
                ("Core Role: The Primary Information Filtering Engine of the Modern Web", 0, True, 18),
                ("Information Overload Problem:", 1, True, 16),
                ("Digital catalogs contain millions of items (Netflix movies, Spotify tracks, Amazon products) far exceeding human evaluation capacity.", 2, False, 15),
                ("Recommenders act as personalized decision-support filters matching users to relevant items.", 2, False, 15),
                ("Business & Economic Significance:", 1, True, 16),
                ("Powers >75% of viewer hours on Netflix, >35% of revenue on Amazon, and >70% of watch time on YouTube.", 2, False, 15),
                ("Directly drives user retention, platform engagement, catalog monetization, and satisfaction.", 2, False, 15),
                ("Technical Mechanism: Neural Collaborative Filtering (NCF / NeuMF):", 1, True, 16),
                ("Learns latent low-dimensional vector representations for users (u ∈ R^d) and items (v ∈ R^d).", 2, False, 15),
                ("Combines linear Generalized Matrix Factorization (GMF) with non-linear Multi-Layer Perceptrons (MLP).", 2, False, 15),
            ]
        },
        # SLIDE 3
        {
            "title": "The Critical Flaws in Current Recommenders (The Dual Crisis)",
            "paragraphs": [
                ("Core Vulnerability: The Interconnected Failures of Modern Recommender Systems", 0, True, 18),
                ("Flaw 1: The Popularity Bias & Long-Tail Starvation Trap", 1, True, 16),
                ("Superstar Economics: The top 1% of head items receive 80%+ of exposure, starving niche/tail items.", 2, False, 15),
                ("Severe Miscalibration: Users with niche tastes are force-fed blockbusters, degrading satisfaction.", 2, False, 15),
                ("Flaw 2: Severe Adversarial Vulnerability (Shilling & Review-Bombing)", 1, True, 16),
                ("Open Data Ingestion: Recommenders naively assume all rating data is genuine and trustworthy.", 2, False, 15),
                ("Malicious actors inject Sybil botnets, 5★ Bandwagon shilling, coordinated 1★ Nuke raids, and AGAS bots.", 2, False, 15),
                ("The Fatal Dilemma of Existing Calibration (Vanilla SUPER Collapse):", 1, True, 16),
                ("Existing debiasing algorithms naively trust raw interaction counts to compute popularity ratios.", 2, False, 15),
                ("Under attack, fake ratings distort volume and quotas, causing calibration to completely collapse!", 2, False, 15),
            ]
        },
        # SLIDE 4
        {
            "title": "Our Solution: The RRFN-LLM-SUPER Architecture Flow",
            "paragraphs": [
                ("Core Concept: Don't Just Calibrate — Denoise, Decouple, and Calibrate Jointly", 0, True, 18),
                ("End-to-End Architectural Pipeline Flow:", 1, True, 16),
                ("Stage 1: Multi-View Tri-Signal Defense (Statistical RRFN + LLM Auditor + Spatiotemporal Bomb Scorer).", 2, False, 15),
                ("Stage 2: Convex Reliability Fusion producing continuous sample weights w(u, i) ∈ [0.02, 1.0].", 2, False, 15),
                ("Stage 3: Reliability-Weighted Pareto Partitioning creating attack-invariant Head and Tail item sets.", 2, False, 15),
                ("Stage 4: Decoupled Dual Training (M_pop and M_tail) using Risk-Consistent Loss with ΔT slack.", 2, False, 15),
                ("Stage 5: Bayesian Popularity Inclination (C̃_u) & Blueprint Denoising pruning poisoned history.", 2, False, 15),
                ("Stage 6: Calibrated Soft Top-N Merging achieving multi-objective Pareto-optimal recommendations.", 2, False, 15),
                ("Why This Architecture Succeeds:", 1, True, 16),
                ("Orthogonal defenses cover each other's blind spots; decoupled neural heads prevent gradient dominance.", 2, False, 15),
            ]
        },
        # SLIDE 5
        {
            "title": "Step 1: Data Ingestion, Filtering & Leakage-Free Temporal Split",
            "paragraphs": [
                ("Core Goal: Ingest MovieLens-1M data, prevent cold-start sparsity, and eliminate look-ahead leakage.", 0, True, 18),
                ("Iterative k-Core Graph Filtering (k ≥ 5):", 1, True, 16),
                ("Prunes users and items with < 5 interactions until convergence to eliminate trivial cold-start artifacts.", 2, False, 15),
                ("Preserves natural power-law catalog distributions and authentic graph connectivity.", 2, False, 15),
                ("Contiguous 0-Indexed Remapping:", 1, True, 16),
                ("Maps arbitrary non-sequential MovieLens user/movie IDs into dense contiguous tensor spaces.", 2, False, 15),
                ("Strict Temporal Leave-One-Out (LOO) Partitioning:", 1, True, 16),
                ("Test Set: The chronological last interaction per user (timestamp N).", 2, False, 15),
                ("Validation Set: The second-to-last interaction per user (timestamp N-1).", 2, False, 15),
                ("Training Set: All preceding interactions (timestamps 1 to N-2).", 2, False, 15),
                ("Zero Look-Ahead Bias: Guarantees future interactions never contaminate training representations.", 2, False, 15),
            ]
        },
        # SLIDE 6
        {
            "title": "Step 2: Threat Modeling & Adversarial Attack Simulation",
            "paragraphs": [
                ("Core Goal: Rigorously stress-test the recommender under 4 realistic adversarial threat models.", 0, True, 18),
                ("Adversarial Attack Simulation Suite (Noise Budget ρ ∈ [5%, 20%]):", 1, True, 16),
                ("1. Random Rating Flip Noise: Simulates noisy data transmission or accidental user rating entry.", 2, False, 15),
                ("2. Bandwagon Shilling: Sybil bot accounts rate popular blockbusters high to build false credibility, while pushing target long-tail items with 5★ ratings.", 2, False, 15),
                ("3. Nuke Review-Bombing: Coordinated bot flood dropping 1★ ratings on popular head items within sliding 24-hour temporal windows to artificially suppress competitor items.", 2, False, 15),
                ("4. AGAS (Agentic Group Shilling - ICDM 2026): Multi-agent bot coordination (mainstream team + niche team) with staggered multi-round profile activation to evade standard statistical filters.", 2, False, 15),
                ("Strict Ground-Truth Integrity:", 1, True, 16),
                ("Only the Training partition is attacked; Validation and Test sets remain 100% clean ground truth.", 2, False, 15),
            ]
        },
        # SLIDE 7
        {
            "title": "Step 3: Tri-Signal Multi-View Reliability Extraction",
            "paragraphs": [
                ("Core Goal: Extract 3 orthogonal trust signals for every interaction before sample fusion.", 0, True, 18),
                ("Why Single Defenses Fail: Statistical tests miss semantic nuance; LLMs are costly; time filters miss slow attacks.", 1, True, 16),
                ("The 3 Complementary Defense Pillars:", 1, True, 16),
                ("1. Statistical Consistency Signal (R_RRFN):", 2, True, 15),
                ("Calculated as the posterior probability ratio P(Y=y_obs | u, i) / max_k P(Y=k | u, i) from warm NeuMF.", 2, False, 15),
                ("2. Semantic Profile Auditor (R_LLM):", 2, True, 15),
                ("Pre-filters to suspicious items (R_RRFN < 0.60) to slash API latency/cost by >95%; queries Gemini 3.6 Flash.", 2, False, 15),
                ("Evaluates user taste coherence against target movie metadata with SQLite prompt caching.", 2, False, 15),
                ("3. Spatiotemporal Anomaly Detector (R_bomb):", 2, True, 15),
                ("Measures temporal burst acceleration A(i, t) via binary search and rating polarity skew S(i, t).", 2, False, 15),
            ]
        },
        # SLIDE 8
        {
            "title": "Step 4: Convex Reliability Fusion & Sample Reweighting",
            "paragraphs": [
                ("Core Goal: Synthesize the 3 orthogonal signals into continuous sample reliability weights w(u, i).", 0, True, 18),
                ("Mathematical Formulation (Convex Combination):", 1, True, 16),
                ("w(u, i) = α · R_RRFN(u, i) + β · R_LLM(u, i) + γ · R_bomb(i, t)", 2, False, 15),
                ("Normalized weights: α + β + γ = 1.0 (Optimal defaults: α = 0.50, β = 0.30, γ = 0.20).", 2, False, 15),
                ("Critical Engineering Feature: Non-Zero Safety Floor (ε_min = 0.02):", 1, True, 16),
                ("w(u, i) = clip(w(u, i), min_weight=0.02, max_weight=1.00).", 2, False, 15),
                ("Why Not Hard Pruning (0.0)? Eliminating edges fragments bipartite user-item graphs and causes vanishing gradients; soft reweighting attenuates attack gradients by 98% while maintaining training stability.", 2, False, 15),
                ("Downstream Role:", 1, True, 16),
                ("w(u, i) guides Pareto partitioning, loss weighting, and blueprint denoising throughout the pipeline.", 2, False, 15),
            ]
        },
        # SLIDE 9
        {
            "title": "Step 5: Noise Transition Modeling & Risk-Consistent Loss",
            "paragraphs": [
                ("Core Goal: Model rating corruption probabilities and train models with forward-corrected risk loss.", 0, True, 18),
                ("Two-Stage Statistical Denoising Mechanism:", 1, True, 16),
                ("1. Anchor Point Discovery: Identifies clean interaction anchors where model posterior confidence ≥ 85%.", 2, False, 15),
                ("2. Empirical Noise Transition Matrix (T̂): Estimates corruption probabilities P(Ỹ = j | Y* = k).", 2, False, 15),
                ("Learnable Regularized Slack Matrix (ΔT):", 1, True, 16),
                ("T_final = Softmax(T̂ + ΔT), allowing the model to adaptively refine transition estimates during training.", 2, False, 15),
                ("Risk-Consistent Loss Formulation with Frobenius Regularization:", 1, True, 16),
                ("L_risk = - 1/|B| ∑_{(u,i) ∈ B} w(u, i) · log([ p(u, i) · T_final ]_{y_tilde}) + λ_Δ ||ΔT||_F^2", 2, False, 15),
                ("Forward correction mathematically inverts label corruption, allowing models to learn clean representations!", 2, False, 15),
            ]
        },
        # SLIDE 10
        {
            "title": "Step 6: Reliability-Weighted Pareto Catalog Partitioning",
            "paragraphs": [
                ("Core Goal: Segment catalog into Head (H) and Tail (T) using attack-invariant effective volume.", 0, True, 18),
                ("The Fatal Vulnerability of Raw Interaction Counts (∑ 1):", 1, True, 16),
                ("In standard systems, sybil botnets inject thousands of fake 5★ ratings to falsely inflate obscure items into Head status.", 2, False, 15),
                ("Reliability-Weighted Effective Volume (V_eff):", 1, True, 16),
                ("V_eff(i) = ∑_{u ∈ U_i} w(u, i) (Weights replace raw counts).", 2, False, 15),
                ("Adversarial ratings with w(u, i) ≈ 0.02 contribute virtually zero volume, neutralizing sybil inflation!", 2, False, 15),
                ("Pareto Catalog Boundary Partitioning (α_pareto = 0.20):", 1, True, 16),
                ("Items sorted in descending order of V_eff(i).", 2, False, 15),
                ("Head Set (H): Top items accounting for the first 20% of cumulative effective volume.", 2, False, 15),
                ("Tail Set (T): Remaining 80% long-tail items (T = I \\ H).", 2, False, 15),
            ]
        },
        # SLIDE 11
        {
            "title": "Step 7: Decoupled Dual Training (M_pop & M_tail Experts)",
            "paragraphs": [
                ("Core Goal: Train two specialized neural expert models to eliminate popularity gradient dominance.", 0, True, 18),
                ("The Gradient Dominance Problem:", 1, True, 16),
                ("Joint neural models suffer severe popularity bias because head items produce frequent, massive gradient updates, completely drowning out delicate long-tail niche representations.", 2, False, 15),
                ("Decoupled Dual Expert Architecture:", 1, True, 16),
                ("1. Head Specialist Model (M_pop): Trained exclusively on interactions (u, i) with i ∈ H.", 2, False, 15),
                ("2. Long-Tail Specialist Model (M_tail): Trained exclusively on interactions (u, i) with i ∈ T.", 2, False, 15),
                ("Risk-Consistent Optimization & Validation:", 1, True, 16),
                ("Both models are optimized with w(u, i)-weighted risk-consistent loss using Adam and Cosine Annealing.", 2, False, 15),
                ("Early stopping evaluated on clean validation data (patience = 3) prevents overfitting.", 2, False, 15),
            ]
        },
        # SLIDE 12
        {
            "title": "Step 8: Bayesian Popularity Inclination & Blueprint Denoising",
            "paragraphs": [
                ("Core Goal: Compute each user's robust popularity taste and build an attack-resistant quota blueprint.", 0, True, 18),
                ("Reliability-Weighted Raw Popularity Ratio:", 1, True, 16),
                ("C_raw(u) = ( ∑_{i ∈ H ∩ I_u} w(u, i) ) / ( ∑_{i ∈ I_u} w(u, i) )", 2, False, 15),
                ("Empirical Bayes Shrinkage Smoothing (τ = 5.0):", 1, True, 16),
                ("C̃(u) = ( ∑ w(u, i) · C_raw(u) + τ · μ_pop ) / ( ∑ w(u, i) + τ )", 2, False, 15),
                ("Smooths sparse cold-start users toward the global prior μ_pop while allowing active users to express true niche tastes.", 2, False, 15),
                ("Denoised Integer Blueprint Construction (b(u) ∈ {0, 1}^K):", 1, True, 16),
                ("Allocates Top-10 slots: k_head = round(K · C̃(u)) Head items + (K - k_head) Long-Tail items.", 2, False, 15),
                ("Prunes low-reliability interactions (w(u, i) < 0.30) to prevent sybil bots from hijacking quota slots!", 2, False, 15),
            ]
        },
        # SLIDE 13
        {
            "title": "Step 9: Calibrated Soft Blueprint Top-N Merging",
            "paragraphs": [
                ("Core Goal: Interleave predictions from M_pop and M_tail following denoised user blueprints.", 0, True, 18),
                ("Top-N Candidate Generation:", 1, True, 16),
                ("M_pop scores all items in H to generate candidate list C_pop(u).", 2, False, 15),
                ("M_tail scores all items in T to generate candidate list C_tail(u).", 2, False, 15),
                ("Soft Blueprint Merging Algorithm:", 1, True, 16),
                ("Iteratively selects the top candidate from C_pop or C_tail based on the sequence defined in b(u).", 2, False, 15),
                ("Multi-Objective Pareto Optimality:", 1, True, 16),
                ("Guarantees that the final Top-10 list exactly matches the user's authentic popularity preference C̃(u) while selecting the highest-relevance items from each expert model.", 2, False, 15),
                ("Completely eliminates popularity distortion without sacrificing ranking accuracy or catalog diversity!", 2, False, 15),
            ]
        },
        # SLIDE 14
        {
            "title": "Experimental Results: Head-to-Head Benchmark (13 Metrics)",
            "paragraphs": [
                ("Core Goal: Evaluate RRFN-LLM-SUPER across all 13 core metrics from the base paper (IEEE Access 2026).", 0, True, 18),
                ("Head-to-Head Comparison (Base Vanilla SUPER under Attack vs. Ours):", 1, True, 16),
                ("Ranking Accuracy: Recall@10 (0.491 vs 0.442, +11.1%) and nDCG@10 (0.384 vs 0.341, +12.6%).", 2, False, 15),
                ("Popularity Calibration Error (RMSE-PC): Slashed from 0.229 down to 0.082 (64.2% error reduction!).", 2, False, 15),
                ("Rank Miscalibration (MRMC): Reduced from 0.224 down to 0.094 (58.1% alignment gain).", 2, False, 15),
                ("Long-Tail Exploration: APLT@10 preserved at 32.4% (vs 24.1%), unique catalog coverage (LTC@10) boosted.", 2, False, 15),
                ("Holistic Composite Metric (GKPI):", 1, True, 16),
                ("GKPI achieves 0.628 vs 0.489 (+28.4% holistic relative gain), outperforming attacked baseline on all 13 metrics!", 2, False, 15),
                ("Adversarial Denoising Precision: Detection F1 > 0.92, ROC-AUC > 0.95 across all noise budgets.", 2, False, 15),
            ]
        },
        # SLIDE 15
        {
            "title": "Ablation Study, Streamlit Dashboard & Conclusion",
            "paragraphs": [
                ("Core Goal: Prove component necessity, provide interactive verification, and summarize contributions.", 0, True, 18),
                ("7-Variant Systematic Ablation Study (Marginal GKPI Degradation When Removed):", 1, True, 16),
                ("Full Model (0.628) → w/o RRFN Risk Loss (-17.5%) → w/o Blueprint Denoising (-17.0%) → w/o Pareto Partitioning (-14.8%) → w/o LLM Auditor (-12.7%) → w/o Bayesian Shrinkage (-9.4%) → w/o Review-Bombing (-8.8%).", 2, False, 15),
                ("Every single module is empirically proven to be mathematically necessary for defense.", 2, False, 15),
                ("Live Interactive Streamlit Dashboard:", 1, True, 16),
                ("Provides real-time simulation, multi-view weight inspection, live LLM prompt auditor, and What-If sandbox.", 2, False, 15),
                ("Summary of Key Contributions:", 1, True, 16),
                ("1. First framework jointly solving adversarial poisoning and popularity miscalibration.", 2, False, 15),
                ("2. Tri-signal multi-view fusion uniting statistical modeling, LLM reasoning, and spatiotemporal bursts.", 2, False, 15),
                ("3. Decoupled dual expert training with attack-invariant Pareto catalog partitioning.", 2, False, 15),
            ]
        }
    ]

    # Process each slide
    layout = prs.slide_layouts[1] # Title and Content
    for idx, data in enumerate(slides_data):
        slide = prs.slides.add_slide(layout)
        
        # 1. Background / Bottom bar (TextBox 4)
        # Position: left=-24606, top=6717256, width=12708000, height=540000
        bottom_bar = slide.shapes.add_textbox(left=-24606, top=6717256, width=12708000, height=540000)
        bottom_bar.name = "TextBox 4"
        bottom_bar.fill.solid()
        bottom_bar.fill.fore_color.rgb = navy_color
        bottom_bar.line.fill.background()

        # 2. Title Shape (Title 1)
        # Left=-24606, Top=-22544, Width=12708000, Height=1260000
        title_shape = slide.shapes.title
        title_shape.name = "Title 1"
        title_shape.left = -24606
        title_shape.top = -22544
        title_shape.width = 12708000
        title_shape.height = 1260000
        title_shape.fill.solid()
        title_shape.fill.fore_color.rgb = navy_color
        title_shape.line.fill.background()

        # Set Title Text
        tf_title = title_shape.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = Inches(0.4)
        tf_title.margin_top = Inches(0.2)
        p_title = tf_title.paragraphs[0]
        p_title.text = f" {data['title']} "
        p_title.font.bold = True
        p_title.font.size = Pt(28 if len(data['title']) < 45 else 24)
        p_title.font.color.rgb = white_color
        p_title.font.name = "Calibri"

        # 3. Logo Picture (Picture 9)
        # Left=9043194, Top=64887, Width=3465103, Height=1085137
        slide.shapes.add_picture('logo_extracted.png', left=9043194, top=64887, width=3465103, height=1085137)

        # 4. Date (Date Placeholder 6)
        date_box = slide.shapes.add_textbox(left=235797, top=6797759, width=2939997, height=383297)
        p_date = date_box.text_frame.paragraphs[0]
        p_date.text = "9/15/2026"
        p_date.font.size = Pt(13)
        p_date.font.color.rgb = white_color

        # 5. Footer (Footer Placeholder 7)
        footer_box = slide.shapes.add_textbox(left=4471194, top=6800056, width=4320000, height=365125)
        p_footer = footer_box.text_frame.paragraphs[0]
        p_footer.text = "BCSE497J Project I - Review II"
        p_footer.font.bold = True
        p_footer.font.size = Pt(14)
        p_footer.font.color.rgb = yellow_color

        # 6. Slide Number (Slide Number Placeholder 8)
        num_box = slide.shapes.add_textbox(left=9424194, top=6797759, width=2939997, height=383297)
        p_num = num_box.text_frame.paragraphs[0]
        p_num.text = str(idx + 1)
        p_num.font.bold = True
        p_num.font.size = Pt(14)
        p_num.font.color.rgb = white_color
        p_num.alignment = PP_ALIGN.RIGHT

        # 7. Content Placeholder
        # Find content placeholder
        content_shape = [s for s in slide.shapes if s.name.startswith("Content Placeholder")][0]
        content_shape.left = 203994
        content_shape.top = 1389856
        content_shape.width = 12192000
        content_shape.height = 5029200
        
        tf_content = content_shape.text_frame
        tf_content.word_wrap = True
        tf_content.margin_left = Inches(0.2)
        tf_content.margin_top = Inches(0.1)

        # Clear default paragraph
        p_first = tf_content.paragraphs[0]
        first_item = data["paragraphs"][0]
        p_first.text = first_item[0]
        p_first.level = first_item[1]
        p_first.font.bold = first_item[2]
        p_first.font.size = Pt(first_item[3])
        p_first.font.name = "Calibri"
        if first_item[1] == 0:
            p_first.font.color.rgb = navy_color

        # Add remaining paragraphs
        for text, level, bold, size in data["paragraphs"][1:]:
            p = tf_content.add_paragraph()
            p.text = text
            p.level = level
            p.font.bold = bold
            p.font.size = Pt(size)
            p.font.name = "Calibri"
            if level == 0:
                p.font.color.rgb = navy_color
            elif level == 1:
                p.font.color.rgb = RGBColor(0x11, 0x18, 0x27)
            else:
                p.font.color.rgb = dark_gray

    output_path = 'BCSE497J Project-I_Review Presentation 2_2026_STEP12345.pptx'
    prs.save(output_path)
    print(f"Successfully generated {len(slides_data)} slides in {output_path}")

if __name__ == "__main__":
    build_presentation()
