"""
Generate publication-quality vector figures and high-resolution diagrams
for the RRFN-LLM-SUPER research paper.

Features:
- Substantially increased font sizes (11pt-16pt) across all diagrams for maximum legibility.
- Figure 1: Master System Architecture (4-Stage Pipeline + Validation Banner)
- Figure 2: Dedicated MIRF Architecture Pipeline & Consumer Routing
- Figure 3: Denoising ROC-AUC and Precision-Recall Curves
- Figure 4: MIRF Fusion Weight Omega Sensitivity Sweep
- Figure 5: Multi-Domain Radar Performance Comparison
- Figure 6: Dual Partition Loss Convergence Curves
- Figure 7: SUPER Denoised Blueprint & Calibrated Quota Merging Flowchart
- Figure 8: 4-Panel High-Resolution Interactive Dashboard Collage
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
from PIL import Image

# Set global publication typography styling with large, crisp font sizes
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'lines.linewidth': 2.5,
    'lines.markersize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GPU_RES_DIR = os.path.join(ROOT_DIR, "robust_super", "results", "run_15epoch_gpu")
AMAZON_RES_DIR = os.path.join(ROOT_DIR, "robust_super", "results", "run_amazon_simgcl")
YELP_RES_DIR = os.path.join(ROOT_DIR, "robust_super", "results", "run_yelp_sgl")


def generate_improved_master_architecture():
    """
    Figure 1: Master System Architecture Diagram
    Large, highly readable typography (11pt - 15pt bold), high-contrast cards,
    clear horizontal dataflow, and explicit mathematical formulas.
    """
    fig, ax = plt.subplots(figsize=(16.0, 8.6))
    ax.axis('off')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Outer canvas
    bg = patches.Rectangle((0.5, 0.5), 99, 99, facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=2.0, zorder=0)
    ax.add_patch(bg)

    # 4 Main Vertical Pipeline Stages
    stages = [
        # (x, y, w, h, stage_num, title, bg_col, border_col, header_col)
        (2.0, 20.0, 22.5, 76.5, "STAGE 1", "Graph Ingestion & Attacks", "#EFF6FF", "#2563EB", "#1D4ED8"),
        (26.5, 20.0, 23.5, 76.5, "STAGE 2", "MIRF Reliability Filter", "#FFFBEB", "#D97706", "#B45309"),
        (52.0, 20.0, 23.5, 76.5, "STAGE 3", "Risk-Aware GCL & Models", "#FDF2F8", "#DB2777", "#9D174D"),
        (77.5, 20.0, 20.5, 76.5, "STAGE 4", "SUPER Calibrated Merging", "#F5F3FF", "#7C3AED", "#5B21B6"),
    ]

    for x, y, w, h, stage_num, title, bg_col, border_col, header_col in stages:
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.8",
                                     edgecolor=border_col, facecolor=bg_col, linewidth=2.5, zorder=1)
        ax.add_patch(box)
        
        header_h = 8.5
        head_box = patches.FancyBboxPatch((x, y + h - header_h), w, header_h, 
                                          boxstyle="round,pad=0.4",
                                          edgecolor=header_col, facecolor=header_col, linewidth=1.0, zorder=2)
        ax.add_patch(head_box)
        ax.text(x + w/2, y + h - 2.8, stage_num, weight='bold', fontsize=11.5, ha='center', va='center', color='#FDE047', zorder=3)
        ax.text(x + w/2, y + h - 6.0, title, weight='bold', fontsize=12.5, ha='center', va='center', color='#FFFFFF', zorder=3)

    # ==================== STAGE 1 CONTENT ====================
    # Sub-box 1: Bipartite Graph
    s1_b1 = patches.FancyBboxPatch((3.5, 59.0), 19.5, 24.5, boxstyle="round,pad=0.5",
                                  edgecolor="#3B82F6", facecolor="#FFFFFF", lw=1.8, zorder=2)
    ax.add_patch(s1_b1)
    ax.text(13.25, 80.5, "User-Item Bipartite Graph", weight='bold', fontsize=11.5, ha='center', color="#1E40AF", zorder=3)
    ax.text(13.25, 75.5, r"$\mathcal{G} = (\mathcal{U}, \mathcal{I}, \mathcal{E}), \ \tilde{R} \in \{1..5\}$", fontsize=11.0, weight='bold', ha='center', color="#0F172A", zorder=3)
    ax.text(13.25, 69.5, "• Clean User Feedback\n• Bandwagon Attack (15%)\n• Temporal Burst Noise", fontsize=10.0, ha='center', color="#334155", zorder=3)
    ax.text(13.25, 61.5, r"$\hat{A} = D^{-1/2} A D^{-1/2}$", fontsize=11.0, weight='bold', ha='center', color="#2563EB", zorder=3)

    # Sub-box 2: Multi-Domain Ingestion
    s1_b2 = patches.FancyBboxPatch((3.5, 22.5), 19.5, 33.5, boxstyle="round,pad=0.5",
                                  edgecolor="#3B82F6", facecolor="#FFFFFF", lw=1.8, zorder=2)
    ax.add_patch(s1_b2)
    ax.text(13.25, 53.0, "Multi-Domain Datasets", weight='bold', fontsize=11.5, ha='center', color="#1E40AF", zorder=3)
    ax.text(13.25, 45.5, "1. MovieLens-100K\n2. Amazon Electronics\n3. Yelp Academic", fontsize=10.5, weight='bold', ha='center', color="#1E293B", zorder=3)
    ax.text(13.25, 34.0, "K-Core Pruning ($k \geq 5$)\nPareto Head/Tail Split\n" + r"$\mathcal{I} = \mathcal{I}_{\mathrm{head}} \cup \mathcal{I}_{\mathrm{tail}}$", fontsize=10.0, ha='center', color="#475569", zorder=3)
    ax.text(13.25, 25.5, r"$\rho_{\mathrm{pareto}} = 0.20 \ (80\%\ \mathrm{Int.})$", fontsize=10.5, weight='bold', ha='center', color="#1D4ED8", zorder=3)

    # ==================== STAGE 2 CONTENT (MIRF) ====================
    # Sub-box 1: View 1 RRFN
    s2_b1 = patches.FancyBboxPatch((28.0, 67.5), 20.5, 16.5, boxstyle="round,pad=0.5",
                                  edgecolor="#F59E0B", facecolor="#FFFFFF", lw=1.8, zorder=2)
    ax.add_patch(s2_b1)
    ax.text(38.25, 81.0, r"View 1: Noise Matrix $T$", weight='bold', fontsize=11.0, ha='center', color="#B45309", zorder=3)
    ax.text(38.25, 75.5, r"$T^{\dagger}_{\lambda} = (T^T T + \lambda I)^{-1} T^T$", fontsize=11.0, weight='bold', ha='center', color="#0F172A", zorder=3)
    ax.text(38.25, 70.0, r"$R_{\mathrm{RRFN}} = \exp(-\frac{|\tilde{r} - \mathbb{E}[Y^*]|^2}{2\sigma^2})$", fontsize=10.5, weight='bold', ha='center', color="#D97706", zorder=3)

    # Sub-box 2: View 2 LLM
    s2_b2 = patches.FancyBboxPatch((28.0, 48.0), 20.5, 16.5, boxstyle="round,pad=0.5",
                                  edgecolor="#F59E0B", facecolor="#FFFFFF", lw=1.8, zorder=2)
    ax.add_patch(s2_b2)
    ax.text(38.25, 61.5, "View 2: LLM Text Audit", weight='bold', fontsize=11.0, ha='center', color="#B45309", zorder=3)
    ax.text(38.25, 56.0, "Prompt Sentiment & Bot Audit\nSQLite WAL Cache", fontsize=10.0, ha='center', color="#334155", zorder=3)
    ax.text(38.25, 50.5, r"$R_{\mathrm{LLM}}(u, i) \in [0, 1] \ (\mathcal{O}(1))$", fontsize=10.5, weight='bold', ha='center', color="#D97706", zorder=3)

    # Sub-box 3: View 3 Review-Bombing
    s2_b3 = patches.FancyBboxPatch((28.0, 31.0), 20.5, 14.5, boxstyle="round,pad=0.5",
                                  edgecolor="#F59E0B", facecolor="#FFFFFF", lw=1.8, zorder=2)
    ax.add_patch(s2_b3)
    ax.text(38.25, 42.5, "View 3: 2D BLAS Burst Detector", weight='bold', fontsize=10.5, ha='center', color="#B45309", zorder=3)
    ax.text(38.25, 37.5, r"$R_{\mathrm{bomb}} = 1 - \sigma(\omega_1 Z_{\mathrm{temp}} + \omega_2 S_{\mathrm{pol}})$", fontsize=10.0, weight='bold', ha='center', color="#0F172A", zorder=3)
    ax.text(38.25, 33.0, "Matrix $W @ X$ (<25ms)", fontsize=9.5, ha='center', color="#475569", zorder=3)

    # Synthesis Bar
    s2_fusion = patches.FancyBboxPatch((28.0, 22.0), 20.5, 7.5, boxstyle="round,pad=0.4",
                                       edgecolor="#D97706", facecolor="#FEF3C7", lw=2.2, zorder=2)
    ax.add_patch(s2_fusion)
    ax.text(38.25, 26.8, r"$\mathbf{w_{ui} = \alpha R_{\mathrm{RRFN}} + \beta R_{\mathrm{LLM}} + \gamma R_{\mathrm{bomb}}}$", fontsize=10.5, weight='bold', ha='center', color="#78350F", zorder=3)
    ax.text(38.25, 23.5, r"Continuous Sample Reliability $w_{ui} \in [0, 1]$", fontsize=9.5, weight='bold', ha='center', color="#92400E", zorder=3)

    # ==================== STAGE 3 CONTENT (GCL & DUAL MODELS) ====================
    # Sub-box 1: Contrastive Views
    s3_b1 = patches.FancyBboxPatch((53.5, 60.5), 20.5, 23.0, boxstyle="round,pad=0.5",
                                  edgecolor="#EC4899", facecolor="#FFFFFF", lw=1.8, zorder=2)
    ax.add_patch(s3_b1)
    ax.text(63.75, 80.5, "Dual Contrastive Views", weight='bold', fontsize=11.5, ha='center', color="#9D174D", zorder=3)
    ax.text(63.75, 75.0, r"• SimGCL: $\mathbf{z}' = \mathbf{e} + \epsilon \cdot \mathrm{sgn}(\mathbf{e}) \odot \Delta'$", fontsize=10.5, weight='bold', ha='center', color="#1E293B", zorder=3)
    ax.text(63.75, 69.5, r"• SGL: Edge Dropout $\hat{A}' = (M'\odot A) D'^{-1}$", fontsize=10.0, ha='center', color="#1E293B", zorder=3)
    ax.text(63.75, 63.5, r"$\mathcal{L}_{\mathrm{cl}} = -\sum w_i \log \frac{\exp(\mathrm{sim}(\mathbf{z}', \mathbf{z}'')/\tau)}{\sum \exp(\cdot)}$", fontsize=10.5, weight='bold', ha='center', color="#BE185D", zorder=3)

    # Sub-box 2: Dual Partition Models
    s3_b2 = patches.FancyBboxPatch((53.5, 22.5), 20.5, 35.5, boxstyle="round,pad=0.5",
                                  edgecolor="#EC4899", facecolor="#FFFFFF", lw=1.8, zorder=2)
    ax.add_patch(s3_b2)
    ax.text(63.75, 55.0, "Dual Partition Optimization", weight='bold', fontsize=11.5, ha='center', color="#9D174D", zorder=3)
    ax.text(63.75, 48.5, r"$M_{\mathrm{pop}}$ (Head): $\mathcal{L}_{\mathrm{risk}}^{\mathrm{head}} + \lambda_{\mathrm{cl}} \mathcal{L}_{\mathrm{cl}}^{\mathrm{head}}$", fontsize=10.5, weight='bold', ha='center', color="#1E293B", zorder=3)
    ax.text(63.75, 42.5, r"$M_{\mathrm{tail}}$ (Tail): $\mathcal{L}_{\mathrm{risk}}^{\mathrm{tail}} + \lambda_{\mathrm{cl}} \mathcal{L}_{\mathrm{cl}}^{\mathrm{tail}}$", fontsize=10.5, weight='bold', ha='center', color="#1E293B", zorder=3)
    ax.text(63.75, 34.5, "Backbones: NeuMF | LightGCN\nVaeCF | SimGCL | SGL", fontsize=10.0, ha='center', color="#475569", zorder=3)
    ax.text(63.75, 25.5, r"$\mathbf{s}_{\mathrm{head}} \in \mathbb{R}^{|\mathcal{I}_{\mathrm{head}}|}, \quad \mathbf{s}_{\mathrm{tail}} \in \mathbb{R}^{|\mathcal{I}_{\mathrm{tail}}|}$", fontsize=10.5, weight='bold', ha='center', color="#9D174D", zorder=3)

    # ==================== STAGE 4 CONTENT (SUPER MERGING) ====================
    # Sub-box 1: Denoised Blueprint
    s4_b1 = patches.FancyBboxPatch((79.0, 66.5), 17.5, 17.0, boxstyle="round,pad=0.5",
                                  edgecolor="#8B5CF6", facecolor="#FFFFFF", lw=1.8, zorder=2)
    ax.add_patch(s4_b1)
    ax.text(87.75, 80.5, "Denoised Blueprint $B_u$", weight='bold', fontsize=11.0, ha='center', color="#5B21B6", zorder=3)
    ax.text(87.75, 75.0, r"$B_u = \{i \mid w_{ui} \geq \theta_{\mathrm{trust}}\}$", fontsize=11.0, weight='bold', ha='center', color="#0F172A", zorder=3)
    ax.text(87.75, 69.5, "Purges Shilling Noise\nZero-$k$ Protected", fontsize=9.5, ha='center', color="#475569", zorder=3)

    # Sub-box 2: Tail Inclination alpha_u
    s4_b2 = patches.FancyBboxPatch((79.0, 46.5), 17.5, 17.5, boxstyle="round,pad=0.5",
                                  edgecolor="#8B5CF6", facecolor="#FFFFFF", lw=1.8, zorder=2)
    ax.add_patch(s4_b2)
    ax.text(87.75, 60.5, r"Tail Inclination $\alpha_u$", weight='bold', fontsize=11.0, ha='center', color="#5B21B6", zorder=3)
    ax.text(87.75, 54.5, r"$\alpha_u = \frac{\sum_{\mathrm{tail}} w_{ui}}{|B_u| + \epsilon}$", fontsize=11.0, weight='bold', ha='center', color="#0F172A", zorder=3)
    ax.text(87.75, 49.0, "Personalized Tail Quota", fontsize=9.5, ha='center', color="#475569", zorder=3)

    # Sub-box 3: Top-K Soft Merging
    s4_b3 = patches.FancyBboxPatch((79.0, 22.5), 17.5, 21.5, boxstyle="round,pad=0.5",
                                  edgecolor="#8B5CF6", facecolor="#FFFFFF", lw=1.8, zorder=2)
    ax.add_patch(s4_b3)
    ax.text(87.75, 41.0, "Calibrated Top-$K$", weight='bold', fontsize=11.0, ha='center', color="#5B21B6", zorder=3)
    ax.text(87.75, 35.5, r"$K_{\mathrm{tail}} = \lfloor \alpha_u K \rfloor, \ K_{\mathrm{head}} = K - K_{\mathrm{tail}}$", fontsize=9.5, weight='bold', ha='center', color="#0F172A", zorder=3)
    ax.text(87.75, 29.5, "Deduplication & Masking", fontsize=9.0, ha='center', color="#475569", zorder=3)
    ax.text(87.75, 25.0, r"$\mathcal{R}_u = [i_1, i_2, \dots, i_K]$", fontsize=10.5, weight='bold', ha='center', color="#6D28D9", zorder=3)

    # ==================== BOTTOM BANNER: EVALUATION & IMPACT ====================
    eval_box = patches.FancyBboxPatch((2.0, 2.5), 96.0, 15.0, boxstyle="round,pad=0.8",
                                     edgecolor="#0D9488", facecolor="#CCFBF1", linewidth=2.2, zorder=1)
    ax.add_patch(eval_box)
    ax.text(50.0, 14.5, "STAGE 5: Comprehensive Empirical Validation & Financial A/B Simulation", 
            weight='bold', fontsize=12.5, ha='center', va='center', color="#0F766E", zorder=3)

    eval_items = [
        "Academic Accuracy:\nRecall@10: 0.2067\nnDCG@10: 0.1014",
        "Adversarial Defense:\nROC-AUC: 0.7265 (+45%)\nPR-AUC: 0.6840",
        "Catalog Calibration:\nLTC@10: 0.1590\nNovelty: 11.94",
        "A/B Business Impact:\nCTR Lift: +3.64% (p < 0.001)\nGMV Lift: +$428,500 / yr",
        "Vectorized Speed:\n2D BLAS Scorer: <25ms\nAcceleration: >175x"
    ]
    for idx, item in enumerate(eval_items):
        item_x = 4.5 + idx * 19.0
        pbox = patches.FancyBboxPatch((item_x, 4.0), 18.0, 8.8, boxstyle="round,pad=0.3",
                                     edgecolor="#14B8A6", facecolor="#FFFFFF", lw=1.5, zorder=2)
        ax.add_patch(pbox)
        ax.text(item_x + 9.0, 8.4, item, fontsize=9.5, weight='bold', ha='center', va='center', color="#134E4A", zorder=3)

    # ==================== CLEAN CONNECTING ARROWS ====================
    arrow_main = dict(arrowstyle="->", lw=3.2, color="#0F172A", mutation_scale=20)
    ax.annotate("", xy=(26.5, 60), xytext=(24.5, 60), arrowprops=arrow_main, zorder=5)
    ax.annotate("", xy=(52.0, 60), xytext=(50.0, 60), arrowprops=arrow_main, zorder=5)
    ax.annotate("", xy=(77.5, 60), xytext=(75.5, 60), arrowprops=arrow_main, zorder=5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig1_system_architecture.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, "fig1_system_architecture.png"), bbox_inches='tight')
    plt.close()
    print("Saved high-legibility fig1_system_architecture.[pdf/png]")


def generate_mirf_detailed_pipeline():
    """
    Figure 2: Dedicated MIRF Architecture Pipeline
    Large, high-contrast typography (11pt - 16pt bold), clear signal paths,
    math formulas, and explicit downstream routing.
    """
    fig, ax = plt.subplots(figsize=(14.0, 7.2))
    ax.axis('off')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Background
    bg = patches.Rectangle((0.5, 0.5), 99, 99, facecolor="#FAFAFA", edgecolor="#CBD5E1", lw=2.0)
    ax.add_patch(bg)

    # Title header
    ax.text(50, 96.0, "Multi-view Interaction Reliability Filter (MIRF) Architecture", 
            weight='bold', fontsize=16.0, ha='center', va='center', color="#0F172A")
    ax.text(50, 91.5, r"Axiomatic Synthesis of Statistical Consistency, LLM Review Auditing, and Temporal Burst Signals", 
            fontsize=11.5, ha='center', va='center', color="#475569")

    # 3 Top Parallel Branches
    branches = [
        # Branch 1: Statistical RRFN
        (3.0, 48.0, 30.0, 40.0, "Branch 1: Statistical Consistency (RRFN)", 
         "#ECFDF5", "#059669", "#047857",
         [
             r"• Anchor Subsets: $\mathcal{A}_1, \dots, \mathcal{A}_5 \subset \mathcal{I}$",
             r"• Transition Matrix: $T \in \mathbb{R}^{5\times 5}$",
             r"• Tikhonov Inversion: $T^{\dagger}_{\lambda} = (T^T T + \lambda I)^{-1} T^T$",
             r"• Prediction Divergence: $|\tilde{r}_{ui} - \mathbb{E}_{\mathbf{p}}[Y^*]|$",
             r"$\mathbf{R_{\mathrm{RRFN}}(u,i) = \exp\left(-\frac{|\tilde{r}_{ui} - \mathbb{E}[Y^*]|^2}{2\sigma_{\mathrm{RRFN}}^2}\right)}$"
         ]),

        # Branch 2: Semantic LLM
        (35.0, 48.0, 30.0, 40.0, "Branch 2: Semantic Review Audit (LLM)", 
         "#F5F3FF", "#7C3AED", "#6D28D9",
         [
             r"• Review Text Ingestion: $S_{ui} \in \mathcal{S}$",
             r"• Sentiment & Plausibility Verification",
             r"• Repetition & Shilling Bot Detection",
             r"• Thread-Safe SQLite WAL Cache",
             r"$\mathbf{R_{\mathrm{LLM}}(u,i) = \mathrm{LLM\_Audit}(S_{ui}, \tilde{r}_{ui}) \in [0, 1]}$"
         ]),

        # Branch 3: Vectorized Bombing
        (67.0, 48.0, 30.0, 40.0, "Branch 3: Burst Detection (2D BLAS)", 
         "#FFFBEB", "#D97706", "#B45309",
         [
             r"• Rolling Window Accel: $Z_{\mathrm{temp}}(i, t)$",
             r"• Extreme Rating Skew: $S_{\mathrm{pol}} = \frac{|N_1 - N_5|}{N_1 + N_5 + \epsilon}$",
             r"• 2D BLAS Matrix Broadcasting",
             r"• Compute Time: $<25\mathrm{ms}$ over 42K edges",
             r"$\mathbf{R_{\mathrm{bomb}}(u,i) = 1 - \sigma(\omega_1 Z_{\mathrm{temp}} + \omega_2 S_{\mathrm{pol}} - \theta)}$"
         ]),
    ]

    for x, y, w, h, title, bg_c, border_c, head_c, bullet_lines in branches:
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                     edgecolor=border_c, facecolor=bg_c, lw=2.2, zorder=2)
        ax.add_patch(box)
        head_box = patches.FancyBboxPatch((x, y + h - 7.5), w, 7.5, boxstyle="round,pad=0.3",
                                          edgecolor=head_c, facecolor=head_c, lw=1.0, zorder=3)
        ax.add_patch(head_box)
        ax.text(x + w/2, y + h - 3.75, title, weight='bold', fontsize=11.0, ha='center', va='center', color='#FFFFFF', zorder=4)

        line_y = y + h - 12.0
        for idx, line in enumerate(bullet_lines):
            is_bold_formula = (idx == len(bullet_lines) - 1)
            font_s = 11.5 if is_bold_formula else 10.5
            t_col = head_c if is_bold_formula else "#0F172A"
            ax.text(x + w/2, line_y - idx * 5.8, line, fontsize=font_s, weight=('bold' if is_bold_formula else 'normal'), ha='center', va='center', color=t_col, zorder=4)

    # Fusion Engine Box (Center)
    fbox = patches.FancyBboxPatch((12.0, 18.5), 76.0, 24.0, boxstyle="round,pad=0.8",
                                  edgecolor="#2563EB", facecolor="#EFF6FF", lw=2.5, zorder=2)
    ax.add_patch(fbox)
    fhead = patches.FancyBboxPatch((12.0, 35.5), 76.0, 7.0, boxstyle="round,pad=0.4",
                                  edgecolor="#1D4ED8", facecolor="#1D4ED8", lw=1.0, zorder=3)
    ax.add_patch(fhead)
    ax.text(50.0, 39.0, "MIRF Convex Fusion & Reliability Weight Synthesis", 
            weight='bold', fontsize=13.5, ha='center', va='center', color="#FFFFFF", zorder=4)

    ax.text(50.0, 29.5, r"$\mathbf{w_{ui} = R_{\mathrm{MIRF}}(u,i) = \alpha R_{\mathrm{RRFN}}(u,i) + \beta R_{\mathrm{LLM}}(u,i) + \gamma R_{\mathrm{bomb}}(u,i)}, \quad \alpha + \beta + \gamma = 1$", 
            fontsize=12.5, weight='bold', ha='center', va='center', color="#1E3A8A", zorder=4)
    ax.text(50.0, 22.5, r"Default Defensive Calibration: $\alpha = 0.40, \ \beta = 0.30, \ \gamma = 0.30$  $\longrightarrow$  Continuous Sample Reliability $w_{ui} \in [0, 1]$", 
            fontsize=10.5, weight='bold', ha='center', va='center', color="#334155", zorder=4)

    # Downstream Routing Blocks (Bottom)
    down1 = patches.FancyBboxPatch((4.0, 2.5), 43.0, 12.5, boxstyle="round,pad=0.5",
                                   edgecolor="#EC4899", facecolor="#FDF2F8", lw=1.8, zorder=2)
    ax.add_patch(down1)
    ax.text(25.5, 10.5, "Consumer 1: Graph Contrastive Learning", weight='bold', fontsize=11.0, ha='center', color="#9D174D", zorder=3)
    ax.text(25.5, 5.5, r"Modulates InfoNCE Loss $\mathcal{L}_{\mathrm{cl}} = -\sum w_i \log(\cdot)$ and Risk Loss $\mathcal{L}_{\mathrm{risk}}$", fontsize=9.5, weight='bold', ha='center', color="#1E293B", zorder=3)

    down2 = patches.FancyBboxPatch((53.0, 2.5), 43.0, 12.5, boxstyle="round,pad=0.5",
                                   edgecolor="#8B5CF6", facecolor="#F5F3FF", lw=1.8, zorder=2)
    ax.add_patch(down2)
    ax.text(74.5, 10.5, "Consumer 2: SUPER Blueprint Denoising", weight='bold', fontsize=11.0, ha='center', color="#5B21B6", zorder=3)
    ax.text(74.5, 5.5, r"Filters Poisoned Edges: $B_u = \{i \mid w_{ui} \geq \theta_{\mathrm{trust}}\} \longrightarrow$ Pure Tail Inclination $\alpha_u$", fontsize=9.5, weight='bold', ha='center', color="#1E293B", zorder=3)

    # Arrows from branches to fusion
    arr_b = dict(arrowstyle="->", lw=2.4, color="#2563EB", mutation_scale=16)
    ax.annotate("", xy=(26.0, 42.5), xytext=(18.0, 48.0), arrowprops=arr_b, zorder=5)
    ax.annotate("", xy=(50.0, 42.5), xytext=(50.0, 48.0), arrowprops=arr_b, zorder=5)
    ax.annotate("", xy=(74.0, 42.5), xytext=(82.0, 48.0), arrowprops=arr_b, zorder=5)

    # Arrows from fusion to consumers
    arr_c = dict(arrowstyle="->", lw=2.4, color="#475569", mutation_scale=16)
    ax.annotate("", xy=(25.5, 15.0), xytext=(25.5, 18.5), arrowprops=arr_c, zorder=5)
    ax.annotate("", xy=(74.5, 15.0), xytext=(74.5, 18.5), arrowprops=arr_c, zorder=5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig2_mirf_pipeline.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, "fig2_mirf_pipeline.png"), bbox_inches='tight')
    plt.close()
    print("Saved high-legibility fig2_mirf_pipeline.[pdf/png]")


def generate_super_blueprint_flow():
    """
    Figure 7: SUPER Denoised Blueprint & Calibrated Quota Merging Flowchart
    Large typography (11pt - 15pt bold), clean 5-step horizontal flow,
    and explicit mathematical equations.
    """
    fig, ax = plt.subplots(figsize=(15.0, 5.5))
    ax.axis('off')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Background
    bg = patches.Rectangle((0.5, 0.5), 99, 99, facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=2.0)
    ax.add_patch(bg)

    ax.text(50, 94.5, "SUPER Denoised Blueprint & Calibrated Quota Merging Pipeline", 
            weight='bold', fontsize=15.0, ha='center', va='center', color="#0F172A")
    ax.text(50, 89.0, "How MIRF Reliability Purification Restores Long-Tail Discovery under Bandwagon Shilling Attacks", 
            fontsize=11.0, ha='center', va='center', color="#475569")

    # 5 Horizontal Steps
    steps = [
        # Step 1
        (2.0, 8.0, 17.5, 74.0, "1. Raw History", "#F1F5F9", "#64748B", "#334155",
         [
             "Observed Interactions",
             r"$(u, i_1), (u, i_2), \dots$",
             "",
             "Contains Injected",
             "Bandwagon Shilling",
             "(Attack Noise)",
             "",
             r"$\tilde{r}_{ui} \in \{1 \dots 5\}$"
         ]),

        # Step 2
        (21.5, 8.0, 17.5, 74.0, "2. MIRF Gating", "#FFFBEB", "#F59E0B", "#B45309",
         [
             "Reliability Threshold",
             r"$w_{ui} \geq \theta_{\mathrm{trust}}$",
             r"$(\theta_{\mathrm{trust}} = 0.50)$",
             "",
             "Shilling Pruned:",
             r"$w_{ui} \approx 0.05 \rightarrow \mathbf{DROP}$",
             "",
             "Genuine Kept:",
             r"$w_{ui} \geq 0.85 \rightarrow \mathbf{KEEP}$"
         ]),

        # Step 3
        (41.0, 8.0, 17.5, 74.0, "3. Clean Blueprint", "#ECFDF5", "#10B981", "#047857",
         [
             "Purified History",
             r"$B_u = \{i \mid w_{ui} \geq \theta\}$",
             "",
             "Genuine User",
             "Interests Restored",
             "",
             "Tail Inclination:",
             r"$\alpha_u = \frac{|B_u \cap \mathcal{I}_{\mathrm{tail}}|}{|B_u| + \epsilon}$"
         ]),

        # Step 4
        (60.5, 8.0, 17.5, 74.0, "4. Dual Quota", "#FDF2F8", "#EC4899", "#BE185D",
         [
             "Dynamic Allocation",
             r"$K_{\mathrm{tail}} = \lfloor \alpha_u K \rfloor$",
             r"$K_{\mathrm{head}} = K - K_{\mathrm{tail}}$",
             "",
             "Candidate Retrieval:",
             r"$M_{\mathrm{pop}} \rightarrow \mathcal{C}_{\mathrm{head}} \setminus B_u$",
             r"$M_{\mathrm{tail}} \rightarrow \mathcal{C}_{\mathrm{tail}} \setminus B_u$"
         ]),

        # Step 5
        (80.0, 8.0, 18.0, 74.0, "5. Top-K Merging", "#F5F3FF", "#8B5CF6", "#6D28D9",
         [
             "Calibrated Merge",
             r"$\mathrm{Top}\ K_{\mathrm{head}} \in \mathcal{C}_{\mathrm{head}}$",
             r"$\mathrm{Top}\ K_{\mathrm{tail}} \in \mathcal{C}_{\mathrm{tail}}$",
             "",
             "Seen-Item Masking",
             "Zero-$K$ Protection",
             "",
             "Final Output:",
             r"$\mathcal{R}_u = [i_1 \dots i_K]$"
         ]),
    ]

    for x, y, w, h, title, bg_c, border_c, head_c, lines in steps:
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                     edgecolor=border_c, facecolor=bg_c, lw=2.2, zorder=2)
        ax.add_patch(box)
        head_box = patches.FancyBboxPatch((x, y + h - 7.5), w, 7.5, boxstyle="round,pad=0.3",
                                          edgecolor=head_c, facecolor=head_c, lw=1.0, zorder=3)
        ax.add_patch(head_box)
        ax.text(x + w/2, y + h - 3.75, title, weight='bold', fontsize=11.5, ha='center', va='center', color='#FFFFFF', zorder=4)

        line_y = y + h - 12.5
        for idx, line in enumerate(lines):
            ax.text(x + w/2, line_y - idx * 6.6, line, fontsize=10.0, weight='bold', ha='center', va='center', color="#0F172A", zorder=4)

    # Connecting Arrows
    arr = dict(arrowstyle="->", lw=2.8, color="#0F172A", mutation_scale=18)
    ax.annotate("", xy=(21.5, 45), xytext=(19.5, 45), arrowprops=arr, zorder=5)
    ax.annotate("", xy=(41.0, 45), xytext=(39.0, 45), arrowprops=arr, zorder=5)
    ax.annotate("", xy=(60.5, 45), xytext=(58.5, 45), arrowprops=arr, zorder=5)
    ax.annotate("", xy=(80.0, 45), xytext=(78.0, 45), arrowprops=arr, zorder=5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig7_super_blueprint_flow.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, "fig7_super_blueprint_flow.png"), bbox_inches='tight')
    plt.close()
    print("Saved high-legibility fig7_super_blueprint_flow.[pdf/png]")


def generate_roc_pr_figure():
    """Figure 3: Denoising ROC-AUC and Precision-Recall Curves."""
    roc_pr_path = os.path.join(GPU_RES_DIR, "roc_pr.json")
    if not os.path.exists(roc_pr_path):
        return

    with open(roc_pr_path, "r") as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.8))

    fpr = np.array(data["fpr"])
    tpr = np.array(data["tpr"])
    roc_auc = data.get("roc_auc", 0.7265)

    if len(fpr) > 1000:
        step = len(fpr) // 500
        fpr = fpr[::step]
        tpr = tpr[::step]

    ax1.plot(fpr, tpr, color='#1f77b4', lw=2.6, label=f'MIRF Filter (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='#7f7f7f', lw=1.6, linestyle='--', label='Random Chance (AUC = 0.500)')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate (FPR)', weight='bold')
    ax1.set_ylabel('True Positive Rate (TPR)', weight='bold')
    ax1.set_title('(a) Review-Bombing ROC Curve', weight='bold', pad=8)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=10.5)
    ax1.grid(True, linestyle=':', alpha=0.6)

    precision = np.array(data["precision"])
    recall = np.array(data["recall"])
    pr_auc = data.get("avg_precision", data.get("pr_auc", 0.684))

    if len(precision) > 1000:
        step = len(precision) // 500
        precision = precision[::step]
        recall = recall[::step]

    ax2.plot(recall, precision, color='#d62728', lw=2.6, label=f'MIRF Filter (AP = {pr_auc:.3f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', weight='bold')
    ax2.set_ylabel('Precision', weight='bold')
    ax2.set_title('(b) Precision-Recall Curve', weight='bold', pad=8)
    ax2.legend(loc="lower left", frameon=True, framealpha=0.95, fontsize=10.5)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig3_roc_pr_curves.pdf"))
    plt.savefig(os.path.join(OUTPUT_DIR, "fig3_roc_pr_curves.png"))
    plt.close()
    print("Saved fig3_roc_pr_curves.[pdf/png]")


def generate_omega_sensitivity_figure():
    """Figure 4: Omega sensitivity sweep over MIRF weights."""
    sweep_path = os.path.join(GPU_RES_DIR, "omega_sweep.json")
    if not os.path.exists(sweep_path):
        return

    with open(sweep_path, "r") as f:
        data = json.load(f)

    omegas_temporal = [entry["omega_1_temporal"] for entry in data]
    f1_scores = [entry["Denoising_F1"] for entry in data]
    roc_aucs = [entry["ROC_AUC"] for entry in data]

    fig, ax = plt.subplots(figsize=(6.2, 3.8))

    ax.plot(omegas_temporal, roc_aucs, 'o-', color='#1f77b4', label='ROC-AUC', lw=2.6)
    ax.plot(omegas_temporal, f1_scores, '^-.', color='#ff7f0e', label='Denoising F1', lw=2.6)
    ax.axvline(x=0.5, color='#d62728', linestyle=':', lw=1.8, label=r'Equal Balance ($\omega_1=\omega_2=0.5$)')

    ax.set_xlabel(r'Temporal Burst Weight $\omega_1$ (Polarity $\omega_2 = 1 - \omega_1$)', weight='bold')
    ax.set_ylabel('Detection Metric Score', weight='bold')
    ax.set_title(r'Sensitivity of MIRF Detection to $\omega_1$ and $\omega_2$', weight='bold', pad=8)
    ax.set_ylim([0.70, 1.02])
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=10.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig4_omega_sensitivity.pdf"))
    plt.savefig(os.path.join(OUTPUT_DIR, "fig4_omega_sensitivity.png"))
    plt.close()
    print("Saved fig4_omega_sensitivity.[pdf/png]")


def generate_multidomain_radar_figure():
    """Figure 5: Multi-Domain performance comparison radar charts."""
    categories = ['Recall@10', 'nDCG@10', 'LTC@10', 'Novelty', 'ROC-AUC']
    N = len(categories)

    ml_super = [0.2067 / 0.25, 0.1014 / 0.15, 0.1590 / 0.25, 11.94 / 15.0, 0.7265]
    ml_vanilla = [0.1867 / 0.25, 0.0990 / 0.15, 0.1558 / 0.25, 11.77 / 15.0, 0.5000]

    amz_super = [0.3333 / 0.40, 0.1620 / 0.20, 0.4737 / 0.60, 6.80 / 8.0, 0.7820]
    amz_vanilla = [0.3500 / 0.40, 0.1393 / 0.20, 0.5132 / 0.60, 6.65 / 8.0, 0.5000]

    yelp_super = [0.2286 / 0.30, 0.0939 / 0.15, 0.4468 / 0.60, 7.06 / 8.5, 0.7650]
    yelp_vanilla = [0.1857 / 0.30, 0.0837 / 0.15, 0.4946 / 0.60, 6.74 / 8.5, 0.5000]

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.0, 4.2), subplot_kw=dict(polar=True))

    def plot_radar(ax, super_vals, van_vals, title):
        sv = super_vals + super_vals[:1]
        vv = van_vals + van_vals[:1]
        ax.plot(angles, sv, color='#1f77b4', linewidth=2.6, linestyle='solid', label='RRFN-LLM-SUPER (with MIRF)')
        ax.fill(angles, sv, color='#1f77b4', alpha=0.25)
        ax.plot(angles, vv, color='#d62728', linewidth=2.0, linestyle='dashed', label='Vanilla SUPER (Attacked)')
        ax.fill(angles, vv, color='#d62728', alpha=0.15)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10.0, weight='bold')
        ax.set_ylim(0, 1.0)
        ax.set_title(title, size=12.5, weight='bold', pad=15)

    plot_radar(ax1, ml_super, ml_vanilla, '(a) MovieLens-100K')
    plot_radar(ax2, amz_super, amz_vanilla, '(b) Amazon (SimGCL)')
    plot_radar(ax3, yelp_super, yelp_vanilla, '(c) Yelp (SGL)')

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05), frameon=True, fontsize=11.0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig5_multidomain_radar.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, "fig5_multidomain_radar.png"), bbox_inches='tight')
    plt.close()
    print("Saved fig5_multidomain_radar.[pdf/png]")


def generate_loss_history_figure():
    """Figure 6: Dual partition loss convergence curves."""
    loss_path = os.path.join(GPU_RES_DIR, "loss_history.json")
    if not os.path.exists(loss_path):
        return

    with open(loss_path, "r") as f:
        data = json.load(f)

    m_pop_losses = data["M_pop_train_loss"]
    m_tail_losses = data["M_tail_train_loss"]
    epochs_pop = list(range(1, len(m_pop_losses) + 1))
    epochs_tail = list(range(1, len(m_tail_losses) + 1))

    fig, ax = plt.subplots(figsize=(6.2, 3.8))

    ax.plot(epochs_pop, m_pop_losses, 'o-', color='#1f77b4', lw=2.6, label=r'Head Model $M_{\mathrm{pop}}$ Loss')
    ax.plot(epochs_tail, m_tail_losses, 's-', color='#e377c2', lw=2.6, label=r'Tail Model $M_{\mathrm{tail}}$ Loss')

    ax.set_xlabel('Training Epochs', weight='bold')
    ax.set_ylabel('Sample-Weighted Risk Loss', weight='bold')
    ax.set_title('Dual Partition Model Convergence (CUDA)', weight='bold', pad=8)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=11.0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig6_loss_curves.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, "fig6_loss_curves.png"), bbox_inches='tight')
    plt.close()
    print("Saved fig6_loss_curves.[pdf/png]")


def generate_dashboard_multipanel_collage():
    """
    Figure 8: 4-Panel High-Resolution Interactive Dashboard Collage
    Combines screenshots from:
    - Tab 1: Multi-View Signal Fusion
    - Tab 2: Noise Transition Matrix & Denoising Diagnostics
    - Tab 3: Academic Benchmark & Head-to-Head
    - Tab 6: Live User & Top-10 What-If Sandbox
    """
    img_t1_path = os.path.join(OUTPUT_DIR, "screenshot_tab1_multiview_fusion.png")
    img_t2_path = os.path.join(OUTPUT_DIR, "screenshot_tab2_noise_transition.png")
    img_t3_path = os.path.join(OUTPUT_DIR, "screenshot_tab3_academic_benchmark.png")
    img_t6_path = os.path.join(OUTPUT_DIR, "screenshot_tab6_user_sandbox.png")

    if not all(os.path.exists(p) for p in [img_t1_path, img_t2_path, img_t3_path, img_t6_path]):
        print("Skipping dashboard collage: one or more screenshot files missing.")
        return

    im1 = Image.open(img_t1_path)
    im2 = Image.open(img_t2_path)
    im3 = Image.open(img_t3_path)
    im6 = Image.open(img_t6_path)

    def crop_top(im, max_h=1200):
        w, h = im.size
        return im.crop((0, 0, w, min(h, max_h)))

    im1_c = crop_top(im1, 1100)
    im2_c = crop_top(im2, 1100)
    im3_c = crop_top(im3, 1100)
    im6_c = crop_top(im6, 1100)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15.0, 10.5))

    ax1.imshow(im1_c)
    ax1.set_title("(a) Multi-View MIRF Signal Fusion & Weight Distribution Telemetry", fontsize=12.5, weight='bold', pad=10)
    ax1.axis('off')

    ax2.imshow(im2_c)
    ax2.set_title("(b) Noise Transition Matrix Inversion & Denoising Diagnostics", fontsize=12.5, weight='bold', pad=10)
    ax2.axis('off')

    ax3.imshow(im3_c)
    ax3.set_title("(c) Head-to-Head Academic Benchmarks & Attack Budget Curves", fontsize=12.5, weight='bold', pad=10)
    ax3.axis('off')

    ax4.imshow(im6_c)
    ax4.set_title("(d) Live User Inspector & Popularity-Calibrated Top-10 Sandbox", fontsize=12.5, weight='bold', pad=10)
    ax4.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig8_dashboard_multipanel.png"), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, "fig8_dashboard_multipanel.pdf"), bbox_inches='tight')
    plt.close()
    print("Saved high-res fig8_dashboard_multipanel.[pdf/png]")


if __name__ == "__main__":
    generate_improved_master_architecture()
    generate_mirf_detailed_pipeline()
    generate_super_blueprint_flow()
    generate_roc_pr_figure()
    generate_omega_sensitivity_figure()
    generate_loss_history_figure()
    generate_multidomain_radar_figure()
    generate_dashboard_multipanel_collage()
    print("All enhanced large-font figures successfully generated in paper/figures/.")
