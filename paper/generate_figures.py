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


def draw_rounded_card(ax, x, y, w, h, bg_color, border_color, lw=1.8, radius=0.8, zorder=1):
    """Draws a crisp rounded card rectangle with subtle border styling."""
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        edgecolor=border_color, facecolor=bg_color,
        linewidth=lw, zorder=zorder
    )
    ax.add_patch(box)
    return box

def draw_header_pill(ax, x, y, w, h, title, subtitle="", bg_color="#1E293B", text_color="#FFFFFF", badge="STAGE", badge_bg="#F59E0B"):
    """Draws an enterprise header pill with a stage badge and dual titles."""
    draw_rounded_card(ax, x, y, w, h, bg_color=bg_color, border_color=bg_color, lw=1.0, radius=0.6, zorder=2)
    badge_w = 4.2
    badge_h = h * 0.65
    badge_x = x + 0.8
    badge_y = y + (h - badge_h) / 2
    draw_rounded_card(ax, badge_x, badge_y, badge_w, badge_h, bg_color=badge_bg, border_color=badge_bg, radius=0.3, zorder=3)
    ax.text(badge_x + badge_w/2, badge_y + badge_h/2, badge, weight='bold', fontsize=9.5, ha='center', va='center', color="#0F172A", zorder=4)
    if subtitle:
        ax.text(x + badge_w + 1.8, y + h * 0.62, title, weight='bold', fontsize=11.5, ha='left', va='center', color=text_color, zorder=4)
        ax.text(x + badge_w + 1.8, y + h * 0.24, subtitle, fontsize=8.8, ha='left', va='center', color="#94A3B8", zorder=4)
    else:
        ax.text(x + badge_w + 1.8, y + h * 0.50, title, weight='bold', fontsize=12.0, ha='left', va='center', color=text_color, zorder=4)

def generate_improved_master_architecture():
    """
    Figure 1: Master System Architecture Diagram
    High-density, publication-grade 5-stage architecture diagram with explicit
    bipartite graph micro-topology, Tikhonov noise inversion heatmap, asynchronous
    LLM review auditor with SQLite WAL cache, 2D BLAS GEMM burst detection,
    risk-aware Graph Contrastive Learning, dual-partition optimization, and
    denoised SUPER blueprint calibrated quota merging.
    Features an orthogonal Reliability Signal Highway with zero line collisions.
    """
    fig, ax = plt.subplots(figsize=(24.0, 13.5))
    ax.axis('off')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Master canvas background
    canvas = patches.Rectangle((0, 0), 100, 100, facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=2.5, zorder=0)
    ax.add_patch(canvas)

    # ==================== MASTER TOP TITLE BANNER ====================
    title_box = patches.Rectangle((0, 94.0), 100, 6.0, facecolor="#0F172A", edgecolor="#0284C7", lw=1.5, zorder=1)
    ax.add_patch(title_box)
    ax.text(50.0, 97.4, "RRFN-LLM-SUPER: Unified End-to-End Robust, Risk-Consistent & Popularity-Calibrated Graph Recommendation Architecture",
            weight='bold', fontsize=15.5, ha='center', va='center', color="#38BDF8", zorder=3)
    ax.text(50.0, 95.2, "Joint Synergistic Ingestion of Adversarial Perturbations, Multi-View Reliability Auditing (MIRF), Risk-Aware Graph Contrastive Learning (GCL), and Calibrated Pareto Debiasing",
            fontsize=10.2, ha='center', va='center', color="#E2E8F0", zorder=3)

    # 5 Major Architectural Columns Definition
    # Y-range for columns: 18.2 to 92.8 (height = 74.6)
    col_y, col_h = 18.2, 74.6
    cols = [
        # (x, y, w, h, stage_num, title, subtitle, bg_col, border_col, head_col, badge_col)
        (1.2, col_y, 18.5, col_h, "STAGE 1", "Graph Ingestion & Attacks", "Multi-Domain Data & Perturbations", "#F0F9FF", "#0284C7", "#0369A1", "#38BDF8"),
        (20.5, col_y, 19.8, col_h, "STAGE 2", "MIRF Reliability Filter", "Tri-View Semantic & Temporal Auditing", "#FFFBEB", "#D97706", "#B45309", "#FBBF24"),
        (41.1, col_y, 18.8, col_h, "STAGE 3", "Risk-Aware GCL Stack", "Contrastive Message Passing & Perturbations", "#FDF2F8", "#DB2777", "#9D174D", "#F472B6"),
        (60.7, col_y, 18.5, col_h, "STAGE 4", "Dual Partition Training", "Tikhonov Risk-Consistent Losses", "#FAF5FF", "#7C3AED", "#5B21B6", "#A78BFA"),
        (80.0, col_y, 18.8, col_h, "STAGE 5", "SUPER Calibrated Merging", "Denoised Blueprint & Quota Allocation", "#F0FDF4", "#16A34A", "#15803D", "#4ADE80"),
    ]

    for x, y, w, h, s_badge, s_title, s_sub, bg_c, bord_c, head_c, badge_c in cols:
        draw_rounded_card(ax, x, y, w, h, bg_color=bg_c, border_color=bord_c, lw=2.2, radius=1.0, zorder=1)
        draw_header_pill(ax, x + 0.4, y + h - 5.2, w - 0.8, 4.8, s_title, s_sub, bg_color=head_c, text_color="#FFFFFF", badge=s_badge, badge_bg=badge_c)

    # =========================================================================
    # STAGE 1: GRAPH INGESTION & ADVERSARIAL ATTACKS (x: 1.2, w: 18.5)
    # =========================================================================
    c1_x, c1_w = 1.8, 17.3
    
    # 1.1 Bipartite Graph Micro-Topology
    draw_rounded_card(ax, c1_x, 61.5, c1_w, 25.0, "#FFFFFF", "#38BDF8", lw=1.6, radius=0.6, zorder=2)
    ax.text(c1_x + c1_w/2, 84.8, "Bipartite Graph Topology $\\mathcal{G} = (\\mathcal{U}, \\mathcal{I}, \\mathcal{E})$", weight='bold', fontsize=10.2, ha='center', color="#0369A1", zorder=3)
    
    # User nodes on left
    u_nodes = [("u_1", 79.5, "#0284C7", "Authentic"),
               ("u_2", 74.5, "#0284C7", "Authentic"),
               ("u_shill", 69.5, "#DC2626", "Bot")]
    for name, y_pos, color, tag in u_nodes:
        circle = patches.Circle((c1_x + 3.0, y_pos), radius=1.35, facecolor=color, edgecolor="#0F172A", lw=1.4, zorder=4)
        ax.add_patch(circle)
        ax.text(c1_x + 3.0, y_pos, name, fontsize=8.2, weight='bold', ha='center', va='center', color="#FFFFFF", zorder=5)
    
    # Item nodes on right (wider rectangles for clean text fit)
    i_nodes = [("i_head", 80.0, "#0D9488"),
               ("i_norm", 74.5, "#475569"),
               ("i_tgt", 69.0, "#EA580C")]
    for name, y_pos, color in i_nodes:
        sq = patches.FancyBboxPatch((c1_x + c1_w - 5.2, y_pos - 1.25), 3.6, 2.5, boxstyle="round,pad=0.0,rounding_size=0.3",
                                    facecolor=color, edgecolor="#0F172A", lw=1.4, zorder=4)
        ax.add_patch(sq)
        ax.text(c1_x + c1_w - 3.4, y_pos, name, fontsize=8.2, weight='bold', ha='center', va='center', color="#FFFFFF", zorder=5)

    # Clean edges (green/blue solid)
    ax.plot([c1_x + 4.35, c1_x + c1_w - 5.2], [79.5, 80.0], color="#16A34A", lw=2.6, zorder=3)
    ax.plot([c1_x + 4.35, c1_x + c1_w - 5.2], [74.5, 74.5], color="#16A34A", lw=2.6, zorder=3)
    ax.plot([c1_x + 4.35, c1_x + c1_w - 5.2], [79.5, 74.5], color="#0284C7", lw=2.0, zorder=3)
    # Attack edges (red dashed / dotted)
    ax.plot([c1_x + 4.35, c1_x + c1_w - 5.2], [69.5, 80.0], color="#DC2626", lw=2.4, linestyle='--', zorder=3)
    ax.plot([c1_x + 4.35, c1_x + c1_w - 5.2], [69.5, 69.0], color="#DC2626", lw=2.6, linestyle=':', zorder=3)
    
    # Legend
    ax.text(c1_x + c1_w/2, 65.5, "Solid Green: Clean Interactions ($r_{ui} \\in [1, 5]$)\nRed Dashed: 15% Bandwagon Camouflage & Push", fontsize=8.2, weight='bold', ha='center', color="#334155", zorder=3)
    ax.text(c1_x + c1_w/2, 62.8, "Normalized Laplacian: $\\hat{A} = D^{-1/2} A D^{-1/2}$", fontsize=9.0, weight='bold', ha='center', color="#0369A1", zorder=3)

    # 1.2 Multi-Domain Benchmark Ingestion
    draw_rounded_card(ax, c1_x, 39.0, c1_w, 21.5, "#FFFFFF", "#38BDF8", lw=1.6, radius=0.6, zorder=2)
    ax.text(c1_x + c1_w/2, 58.8, "Multi-Domain Benchmark Ingestion", weight='bold', fontsize=10.0, ha='center', color="#0369A1", zorder=3)
    
    domains = [
        ("MovieLens-1M Sub-Graph", "6,040 U / 3,706 I / 1M Ratings", "15% Bandwagon Attack (\\beta = 0)"),
        ("Amazon Electronics", "E-Comm Graph with Review Text", "Sparse Long-Tail Skew (\\beta > 0)"),
        ("Yelp Academic", "Local Graph with Text + Timestamps", "Temporal Review-Bombing Bursts")
    ]
    for idx, (d_name, d_stat, d_note) in enumerate(domains):
        d_y = 54.0 - idx * 6.2
        draw_rounded_card(ax, c1_x + 0.6, d_y - 2.0, c1_w - 1.2, 5.2, "#F8FAFC", "#BAE6FD", lw=1.0, radius=0.4, zorder=3)
        ax.text(c1_x + 1.2, d_y + 1.7, f"• {d_name}", weight='bold', fontsize=8.6, ha='left', color="#0F172A", zorder=4)
        ax.text(c1_x + 1.2, d_y + 0.1, f"  {d_stat}", fontsize=8.0, ha='left', color="#475569", zorder=4)
        ax.text(c1_x + 1.2, d_y - 1.3, f"  Challenge: {d_note}", fontsize=7.8, weight='bold', ha='left', color="#0369A1", zorder=4)

    # 1.3 Pareto Catalog Stratification Card
    draw_rounded_card(ax, c1_x, 19.2, c1_w, 18.8, "#FFFFFF", "#38BDF8", lw=1.6, radius=0.6, zorder=2)
    ax.text(c1_x + c1_w/2, 36.3, "Pareto Catalog Stratification (80/20)", weight='bold', fontsize=10.0, ha='center', color="#0369A1", zorder=3)
    
    bar_w = c1_w - 2.0
    draw_rounded_card(ax, c1_x + 1.0, 30.5, bar_w * 0.28, 4.0, "#0284C7", "#0284C7", radius=0.3, zorder=3)
    ax.text(c1_x + 1.0 + (bar_w * 0.28)/2, 32.5, "Head (20%)", fontsize=8.6, weight='bold', ha='center', va='center', color="#FFFFFF", zorder=4)
    draw_rounded_card(ax, c1_x + 1.0 + bar_w * 0.28 + 0.3, 30.5, bar_w * 0.72 - 0.3, 4.0, "#94A3B8", "#94A3B8", radius=0.3, zorder=3)
    ax.text(c1_x + 1.0 + bar_w * 0.28 + 0.3 + (bar_w * 0.72 - 0.3)/2, 32.5, "Long-Tail Niche Catalog (80%)", fontsize=8.6, weight='bold', ha='center', va='center', color="#FFFFFF", zorder=4)
    
    ax.text(c1_x + c1_w/2, 27.6, "$\\mathcal{I} = \\mathcal{I}_{\\mathrm{head}} \\cup \\mathcal{I}_{\\mathrm{tail}}, \\quad \\rho = 0.20$", fontsize=9.6, weight='bold', ha='center', color="#0F172A", zorder=3)
    ax.text(c1_x + c1_w/2, 23.8, "• Head: 20% items consume 80% volume\n• Tail: 80% items starved of gradient signal", fontsize=8.4, ha='center', color="#334155", zorder=3)
    ax.text(c1_x + c1_w/2, 20.4, "Enables decoupled dual-partition modeling", fontsize=8.2, style='italic', weight='bold', ha='center', color="#0284C7", zorder=3)

    # =========================================================================
    # STAGE 2: MIRF MULTI-VIEW RELIABILITY FILTER (x: 20.5, w: 19.8)
    # =========================================================================
    c2_x, c2_w = 21.0, 18.8
    
    # 2.1 View 1: Noise Matrix T & Tikhonov Inversion
    draw_rounded_card(ax, c2_x, 69.5, c2_w, 17.0, "#FFFFFF", "#F59E0B", lw=1.6, radius=0.6, zorder=2)
    ax.text(c2_x + c2_w/2, 84.6, "View 1: Noise Transition & Tikhonov", weight='bold', fontsize=10.0, ha='center', color="#B45309", zorder=3)
    
    # 5x5 Heatmap matrix T
    mat_x = c2_x + 0.8
    mat_y = 79.5
    for r in range(5):
        for c in range(5):
            val_col = "#FEF3C7" if r != c else "#D97706"
            txt_col = "#78350F" if r != c else "#FFFFFF"
            patches_rect = patches.Rectangle((mat_x + c*0.95, mat_y - r*0.95), 0.85, 0.85, facecolor=val_col, edgecolor="#B45309", lw=0.5, zorder=3)
            ax.add_patch(patches_rect)
            if r == c:
                ax.text(mat_x + c*0.95 + 0.425, mat_y - r*0.95 + 0.425, "0.8", fontsize=6.8, weight='bold', ha='center', va='center', color=txt_col, zorder=4)
    
    ax.text(c2_x + 6.2, 80.2, "Anchor Sets: $\\mathcal{A}_1, \\dots, \\mathcal{A}_5 \\subset \\mathcal{I}$", fontsize=8.6, weight='bold', ha='left', color="#0F172A", zorder=3)
    ax.text(c2_x + 6.2, 77.4, "$T_{jk} = P(\\tilde{Y}=k \\mid Y^*=j)$", fontsize=8.8, weight='bold', ha='left', color="#B45309", zorder=3)
    ax.text(c2_x + 6.2, 74.4, "$T^{\\dagger}_{\\lambda} = (T^T T + \\lambda I)^{-1} T^T$", fontsize=9.2, weight='bold', ha='left', color="#0F172A", zorder=3)
    ax.text(c2_x + 6.2, 71.4, "Bound: $\\|T^{\\dagger}_{\\lambda}\\|_2 \\leq 1/(2\\sqrt{\\lambda})$ (Thm 1)", fontsize=8.4, weight='bold', ha='left', color="#78350F", zorder=3)

    # 2.2 View 2: Asynchronous LLM Review Auditor
    draw_rounded_card(ax, c2_x, 52.0, c2_w, 16.2, "#FFFFFF", "#F59E0B", lw=1.6, radius=0.6, zorder=2)
    ax.text(c2_x + c2_w/2, 66.2, "View 2: Asynchronous LLM Review Auditor", weight='bold', fontsize=10.0, ha='center', va='center', color="#B45309", zorder=3)
    
    # Left box: Review Prompt
    draw_rounded_card(ax, c2_x + 0.8, 55.8, 7.6, 7.8, "#F8FAFC", "#CBD5E1", lw=1.1, radius=0.4, zorder=3)
    ax.text(c2_x + 4.6, 61.8, "Review Prompt", weight='bold', fontsize=8.6, ha='center', va='center', color="#0F172A", zorder=4)
    ax.text(c2_x + 4.6, 58.5, "• Sentiment Skew\n• Shill Bot Pattern\n• Semantic Discrepancy", fontsize=7.8, ha='center', va='center', color="#334155", zorder=4)
    
    # Arrow
    ax.annotate("", xy=(c2_x + 9.4, 59.7), xytext=(c2_x + 8.6, 59.7),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#D97706"), zorder=4)
    
    # Right box: LLM Engine
    draw_rounded_card(ax, c2_x + 9.6, 55.8, 8.4, 7.8, "#FFFBEB", "#F59E0B", lw=1.1, radius=0.4, zorder=3)
    ax.text(c2_x + 13.8, 61.8, "LLM Engine", weight='bold', fontsize=8.8, ha='center', va='center', color="#B45309", zorder=4)
    ax.text(c2_x + 13.8, 58.5, "SQLite WAL Cache\n(>85% Hits, $\\mathcal{O}(1)$)\nAsynchronous Thread", fontsize=7.8, ha='center', va='center', color="#78350F", zorder=4)
    
    ax.text(c2_x + c2_w/2, 53.6, "$R_{\\mathrm{LLM}}(u,i) = \\mathrm{LLMAudit}(\\mathrm{Review}_{ui}, r_{ui}) \\in [0, 1]$", fontsize=8.8, weight='bold', ha='center', va='center', color="#0F172A", zorder=3)

    # 2.3 View 3: 2D Level-3 BLAS Burst Detector
    draw_rounded_card(ax, c2_x, 34.8, c2_w, 15.8, "#FFFFFF", "#F59E0B", lw=1.6, radius=0.6, zorder=2)
    ax.text(c2_x + c2_w/2, 48.6, "View 3: 2D BLAS Review-Bomb Detector", weight='bold', fontsize=10.0, ha='center', color="#B45309", zorder=3)
    
    ax.text(c2_x + 1.2, 45.2, "• Temporal Window Density: $Z_{\\mathrm{temp}} = (|\\mathcal{E}_{\\Delta}| - \\mu)/\\sigma$", fontsize=8.2, ha='left', color="#0F172A", zorder=3)
    ax.text(c2_x + 1.2, 42.4, "• Polarity Skew Deviation: $S_{\\mathrm{pol}} = |\\bar{r}_{\\Delta} - \\bar{r}_{\\mathrm{hist}}|$", fontsize=8.2, ha='left', color="#0F172A", zorder=3)
    
    draw_rounded_card(ax, c2_x + 0.8, 36.2, c2_w - 1.6, 4.8, "#FEF3C7", "#D97706", lw=1.0, radius=0.4, zorder=3)
    ax.text(c2_x + c2_w/2, 39.4, "Vectorized 2D BLAS GEMM: $W \\cdot [Z, S]^T - \\theta$", weight='bold', fontsize=8.4, ha='center', color="#78350F", zorder=4)
    ax.text(c2_x + c2_w/2, 37.4, "4.95 ms for 42,000 interactions (>500x Speedup)", fontsize=8.0, weight='bold', ha='center', color="#B45309", zorder=4)

    # 2.4 Convex Reliability Fusion Hub
    draw_rounded_card(ax, c2_x, 19.2, c2_w, 14.5, "#FEF3C7", "#D97706", lw=2.2, radius=0.6, zorder=2)
    ax.text(c2_x + c2_w/2, 31.8, "Convex Reliability Fusion Hub", weight='bold', fontsize=10.5, ha='center', color="#78350F", zorder=3)
    ax.text(c2_x + c2_w/2, 28.5, "$\\mathbf{w_{ui} = \\alpha R_{\\mathrm{RRFN}} + \\beta R_{\\mathrm{LLM}} + \\gamma R_{\\mathrm{bomb}}}$", weight='bold', fontsize=10.2, ha='center', color="#0F172A", zorder=3)
    ax.text(c2_x + c2_w/2, 25.2, "$\\alpha + \\beta + \\gamma = 1.0, \\quad w_{ui} \\in [0, 1]$", fontsize=9.0, weight='bold', ha='center', color="#B45309", zorder=3)
    ax.text(c2_x + c2_w/2, 21.2, "Continuous trust weight modulating downstream\nGCL contrastive loss, Risk Loss, and Blueprint", fontsize=8.2, ha='center', color="#451A03", zorder=3)

    # =========================================================================
    # STAGE 3: RISK-AWARE GRAPH CONTRASTIVE LEARNING (x: 41.1, w: 18.8)
    # =========================================================================
    c3_x, c3_w = 41.7, 17.6
    
    # 3.1 Neural GNN Backbone & Layer Aggregation
    draw_rounded_card(ax, c3_x, 69.5, c3_w, 17.0, "#FFFFFF", "#EC4899", lw=1.6, radius=0.6, zorder=2)
    ax.text(c3_x + c3_w/2, 84.6, "Multi-Layer GNN Neighborhood Smoothing", weight='bold', fontsize=9.8, ha='center', color="#9D174D", zorder=3)
    
    gnn_layers = ["Layer 0:\n$\\mathbf{e}^{(0)}$", "Layer 1:\n$\\mathbf{e}^{(1)}$", "Layer 2:\n$\\mathbf{e}^{(2)}$"]
    for l_idx, l_name in enumerate(gnn_layers):
        lx = c3_x + 1.2 + l_idx * 5.2
        draw_rounded_card(ax, lx, 76.5, 4.4, 5.5, "#FDF2F8", "#F472B6", lw=1.0, radius=0.4, zorder=3)
        ax.text(lx + 2.2, 79.2, l_name, fontsize=8.2, weight='bold', ha='center', va='center', color="#831843", zorder=4)
        if l_idx < 2:
            ax.annotate("", xy=(lx + 4.9, 79.2), xytext=(lx + 4.5, 79.2),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color="#DB2777"), zorder=4)
            
    ax.text(c3_x + c3_w/2, 73.2, "$\\mathbf{e}_u^{(l+1)} = \\sum_{i \\in \\mathcal{N}_u} \\frac{1}{\\sqrt{d_u d_i}} \\mathbf{e}_i^{(l)}$", fontsize=9.2, weight='bold', ha='center', color="#0F172A", zorder=3)
    ax.text(c3_x + c3_w/2, 70.8, "Backbones: LightGCN | SimGCL | SGL | NeuMF", fontsize=8.4, weight='bold', ha='center', color="#475569", zorder=3)

    # 3.2 Dual Contrastive Views: SimGCL & SGL
    draw_rounded_card(ax, c3_x, 47.5, c3_w, 20.5, "#FFFFFF", "#EC4899", lw=1.6, radius=0.6, zorder=2)
    ax.text(c3_x + c3_w/2, 66.0, "Dual Contrastive Augmentations", weight='bold', fontsize=10.0, ha='center', color="#9D174D", zorder=3)
    
    # Branch A: SimGCL
    draw_rounded_card(ax, c3_x + 0.8, 57.0, c3_w - 1.6, 7.5, "#FDF2F8", "#F472B6", lw=1.0, radius=0.4, zorder=3)
    ax.text(c3_x + 1.5, 62.5, "View A (SimGCL): Uniform Latent Noise", weight='bold', fontsize=8.6, ha='left', color="#9D174D", zorder=4)
    ax.text(c3_x + 1.5, 59.2, "$\\mathbf{z}' = \\mathbf{e} + \\epsilon \\cdot \\mathrm{sgn}(\\mathbf{e}) \\odot \\Delta', \\quad \\Delta' \\sim \\mathcal{U}(0, 1)$", fontsize=8.4, weight='bold', ha='left', color="#0F172A", zorder=4)
    
    # Branch B: SGL
    draw_rounded_card(ax, c3_x + 0.8, 48.5, c3_w - 1.6, 7.5, "#FDF2F8", "#F472B6", lw=1.0, radius=0.4, zorder=3)
    ax.text(c3_x + 1.5, 54.0, "View B (SGL): Edge Dropout Masking", weight='bold', fontsize=8.6, ha='left', color="#9D174D", zorder=4)
    ax.text(c3_x + 1.5, 50.7, "$\\hat{A}'' = (M'' \\odot A) D''^{-1}, \\quad M'' \\sim \\mathrm{Bernoulli}(1 - p)$", fontsize=8.4, weight='bold', ha='left', color="#0F172A", zorder=4)

    # 3.3 Reliability-Weighted InfoNCE Loss Engine
    draw_rounded_card(ax, c3_x, 19.2, c3_w, 26.8, "#FFFFFF", "#EC4899", lw=2.0, radius=0.6, zorder=2)
    ax.text(c3_x + c3_w/2, 43.8, "Sample-Weighted InfoNCE Loss", weight='bold', fontsize=10.0, ha='center', color="#9D174D", zorder=3)
    
    draw_rounded_card(ax, c3_x + 0.8, 30.5, c3_w - 1.6, 11.5, "#FDF2F8", "#DB2777", lw=1.2, radius=0.4, zorder=3)
    ax.text(c3_x + c3_w/2, 38.5, "$\\mathcal{L}_{\\mathrm{cl}}(u) = -\\sum_{i \\in \\mathcal{N}_u} \\mathbf{w_{ui}} \\log \\frac{\\exp(\\mathrm{sim}(\\mathbf{z}'_i, \\mathbf{z}''_i)/\\tau)}{\\sum_{j \\in \\mathcal{B}} \\exp(\\mathrm{sim}(\\mathbf{z}'_i, \\mathbf{z}''_j)/\\tau)}$",
            fontsize=8.5, weight='bold', ha='center', color="#831843", zorder=4)
    ax.text(c3_x + c3_w/2, 33.2, "Modulated by MIRF Sample Trust Weight $w_{ui}$", fontsize=8.2, weight='bold', ha='center', color="#475569", zorder=4)
    
    ax.text(c3_x + c3_w/2, 27.2, "• Poisoned Edges ($w_{ui} \\to 0$): Gradient cancelled\n• Authentic Edges ($w_{ui} \\to 1$): Representation pulled\n• Temperature parameter: $\\tau = 0.15$", fontsize=8.2, ha='center', color="#334155", zorder=3)
    ax.text(c3_x + c3_w/2, 21.0, "Aligns embeddings without attack corruption", fontsize=8.4, style='italic', weight='bold', ha='center', color="#DB2777", zorder=3)

    # =========================================================================
    # STAGE 4: DUAL PARTITION RISK OPTIMIZATION (x: 60.7, w: 18.5)
    # =========================================================================
    c4_x, c4_w = 61.2, 17.5
    
    # 4.1 Decoupled Dual-Partition Neural Models
    draw_rounded_card(ax, c4_x, 68.0, c4_w, 18.5, "#FFFFFF", "#7C3AED", lw=1.6, radius=0.6, zorder=2)
    ax.text(c4_x + c4_w/2, 84.6, "Decoupled Dual-Partition Models", weight='bold', fontsize=10.0, ha='center', color="#5B21B6", zorder=3)
    
    draw_rounded_card(ax, c4_x + 0.8, 76.5, c4_w - 1.6, 6.8, "#FAF5FF", "#A78BFA", lw=1.0, radius=0.4, zorder=3)
    ax.text(c4_x + 1.4, 81.2, "Head Expert: $M_{\\mathrm{pop}}(\\Theta_{\\mathrm{head}})$", weight='bold', fontsize=8.6, ha='left', color="#5B21B6", zorder=4)
    ax.text(c4_x + 1.4, 78.2, "Specialized on $\\mathcal{I}_{\\mathrm{head}}$ (Top 20%) $\\to$ Score $\\mathbf{s}_{\\mathrm{head}}$", fontsize=8.0, ha='left', color="#4C1D95", zorder=4)
    
    draw_rounded_card(ax, c4_x + 0.8, 69.0, c4_w - 1.6, 6.8, "#FAF5FF", "#A78BFA", lw=1.0, radius=0.4, zorder=3)
    ax.text(c4_x + 1.4, 73.7, "Tail Expert: $M_{\\mathrm{tail}}(\\Theta_{\\mathrm{tail}})$", weight='bold', fontsize=8.6, ha='left', color="#5B21B6", zorder=4)
    ax.text(c4_x + 1.4, 70.7, "Specialized on $\\mathcal{I}_{\\mathrm{tail}}$ (Niche 80%) $\\to$ Score $\\mathbf{s}_{\\mathrm{tail}}$", fontsize=8.0, ha='left', color="#4C1D95", zorder=4)

    # 4.2 Forward-Corrected Risk-Consistent Loss
    draw_rounded_card(ax, c4_x, 38.5, c4_w, 28.0, "#FFFFFF", "#7C3AED", lw=1.8, radius=0.6, zorder=2)
    ax.text(c4_x + c4_w/2, 64.5, "Tikhonov Risk-Consistent Objective", weight='bold', fontsize=10.0, ha='center', color="#5B21B6", zorder=3)
    
    draw_rounded_card(ax, c4_x + 0.8, 50.0, c4_w - 1.6, 12.5, "#FAF5FF", "#7C3AED", lw=1.2, radius=0.4, zorder=3)
    ax.text(c4_x + c4_w/2, 58.5, "$\\mathcal{L}_{\\mathrm{risk}} = -\\sum_{(u,i)} w_{ui} \\log [T^{\\dagger}_{\\lambda} \\cdot \\hat{P}_{\\theta}(y \\mid u,i)]_{\\tilde{y}_{ui}}$",
            fontsize=8.5, weight='bold', ha='center', color="#4C1D95", zorder=4)
    ax.text(c4_x + c4_w/2, 53.5, "$+ \\mu \\|\\Delta T\\|_F^2 + \\lambda_{\\mathrm{reg}} \\|\\Theta\\|_2^2$",
            fontsize=8.5, weight='bold', ha='center', color="#4C1D95", zorder=4)
    
    ax.text(c4_x + c4_w/2, 46.5, "Theoretical Guarantees (Theorems 1 & 2):", fontsize=8.4, weight='bold', ha='center', color="#0F172A", zorder=3)
    ax.text(c4_x + c4_w/2, 42.0, "• Spectral Bound: $\\|T^{\\dagger}_{\\lambda}\\|_2 \\leq 1 / (2\\sqrt{\\lambda})$ (Smoothness)\n• Asymptotic Risk: $\\mathcal{R}_{\\mathrm{clean}}(\\hat{f}_N) \\to \\mathcal{R}^*$ as $N \\to \\infty$", fontsize=8.2, ha='center', color="#334155", zorder=3)

    # 4.3 Joint Multi-Task Optimization Box
    draw_rounded_card(ax, c4_x, 19.2, c4_w, 17.8, "#FAF5FF", "#7C3AED", lw=1.6, radius=0.6, zorder=2)
    ax.text(c4_x + c4_w/2, 34.5, "Joint Multi-Task Optimization", weight='bold', fontsize=9.8, ha='center', color="#5B21B6", zorder=3)
    ax.text(c4_x + c4_w/2, 30.5, "$\\mathcal{L}_{\\mathrm{total}} = \\mathcal{L}_{\\mathrm{risk}} + \\lambda_{\\mathrm{cl}} \\mathcal{L}_{\\mathrm{cl}}$", weight='bold', fontsize=10.2, ha='center', color="#0F172A", zorder=3)
    ax.text(c4_x + c4_w/2, 25.5, "Dual-branch AdamW Backprop:\n$\\Theta_{\\mathrm{head}} \\leftarrow \\nabla \\mathcal{L}_{\\mathrm{head}}, \\quad \\Theta_{\\mathrm{tail}} \\leftarrow \\nabla \\mathcal{L}_{\\mathrm{tail}}$", fontsize=8.2, ha='center', color="#334155", zorder=3)
    ax.text(c4_x + c4_w/2, 20.8, "Outputs calibrated candidate rankings", fontsize=8.0, style='italic', weight='bold', ha='center', color="#6D28D9", zorder=3)

    # =========================================================================
    # STAGE 5: SUPER CALIBRATED MERGING & RECOMMENDATIONS (x: 80.0, w: 18.8)
    # =========================================================================
    c5_x, c5_w = 80.6, 17.6
    
    # 5.1 Denoised Interaction Blueprint Card
    draw_rounded_card(ax, c5_x, 69.5, c5_w, 17.0, "#FFFFFF", "#16A34A", lw=1.6, radius=0.6, zorder=2)
    ax.text(c5_x + c5_w/2, 84.6, "Denoised Blueprint Filtering $\\mathcal{B}_u$", weight='bold', fontsize=10.0, ha='center', color="#15803D", zorder=3)
    ax.text(c5_x + c5_w/2, 80.5, "$\\mathcal{B}_u = \\{ i \\in \\mathcal{N}_u \\mid w_{ui} \\geq \\theta_{\\mathrm{trust}} \\}$", fontsize=9.4, weight='bold', ha='center', color="#0F172A", zorder=3)
    ax.text(c5_x + c5_w/2, 75.5, "• Prunes fake shilling edges injected by bots\n• Authentic retention rate: $99.89\\%$\n• Dynamic zero-$k$ fallback for cold-start users", fontsize=8.2, ha='center', color="#334155", zorder=3)
    ax.text(c5_x + c5_w/2, 71.0, "Guarantees uncorrupted user taste profile", fontsize=8.2, weight='bold', style='italic', ha='center', color="#16A34A", zorder=3)

    # 5.2 Personalized Tail Inclination Quota
    draw_rounded_card(ax, c5_x, 50.0, c5_w, 18.0, "#FFFFFF", "#16A34A", lw=1.6, radius=0.6, zorder=2)
    ax.text(c5_x + c5_w/2, 65.8, "Personalized Tail Inclination $\\alpha_u$", weight='bold', fontsize=10.0, ha='center', color="#15803D", zorder=3)
    
    draw_rounded_card(ax, c5_x + 0.8, 56.5, c5_w - 1.6, 7.5, "#F0FDF4", "#4ADE80", lw=1.0, radius=0.4, zorder=3)
    ax.text(c5_x + c5_w/2, 61.5, "$\\alpha_u = \\frac{\\sum_{i \\in \\mathcal{B}_u \\cap \\mathcal{I}_{\\mathrm{tail}}} w_{ui}}{|\\mathcal{B}_u| + \\epsilon} \\in [0, 1]$",
            fontsize=8.8, weight='bold', ha='center', color="#166534", zorder=4)
    ax.text(c5_x + c5_w/2, 58.0, "Weighted Tail History Ratio", fontsize=7.8, ha='center', color="#15803D", zorder=4)
    
    ax.text(c5_x + c5_w/2, 53.0, "Dynamic Top-$K$ Budget Allocation:\n$K_{\\mathrm{tail}} = \\lfloor \\alpha_u K \\rfloor, \\quad K_{\\mathrm{head}} = K - K_{\\mathrm{tail}}$", fontsize=8.4, weight='bold', ha='center', color="#0F172A", zorder=3)

    # 5.3 Calibrated Recommendation Slate Assembly
    draw_rounded_card(ax, c5_x, 19.2, c5_w, 29.3, "#F0FDF4", "#16A34A", lw=2.2, radius=0.6, zorder=2)
    ax.text(c5_x + c5_w/2, 46.2, "Calibrated Recommendation Slate", weight='bold', fontsize=10.0, ha='center', color="#15803D", zorder=3)
    
    # Recommendation items
    rec_items = [
        ("1. Head Rec (Score: 0.94)", "#0284C7", "Pop"),
        ("2. Head Rec (Score: 0.91)", "#0284C7", "Pop"),
        ("3. Tail Rec (Score: 0.88)", "#16A34A", "Tail"),
        ("4. Tail Rec (Score: 0.85)", "#16A34A", "Tail"),
    ]
    for r_idx, (r_txt, r_col, r_tag) in enumerate(rec_items):
        ry = 42.0 - r_idx * 2.7
        draw_rounded_card(ax, c5_x + 1.0, ry - 1.0, c5_w - 2.0, 2.4, "#F8FAFC", r_col, lw=0.9, radius=0.3, zorder=4)
        ax.text(c5_x + 1.8, ry + 0.2, r_txt, fontsize=7.8, weight='bold', ha='left', color="#0F172A", zorder=5)
        draw_rounded_card(ax, c5_x + c5_w - 3.8, ry - 0.7, 2.4, 1.8, r_col, r_col, radius=0.2, zorder=5)
        ax.text(c5_x + c5_w - 2.6, ry + 0.2, r_tag, fontsize=7.2, weight='bold', ha='center', va='center', color="#FFFFFF", zorder=6)

    ax.text(c5_x + c5_w/2, 29.5, "$g$-LTC & STER Pareto Calibration", fontsize=8.6, weight='bold', ha='center', color="#166534", zorder=3)
    ax.text(c5_x + c5_w/2, 25.5, "• Deduplication & Score Re-normalization\n• SPAM Exposure Drop (STER@10): $88\\% \\to 72\\%$\n• Genuine Long-Tail Discovery Restored", fontsize=8.0, ha='center', color="#334155", zorder=3)
    ax.text(c5_x + c5_w/2, 20.6, "Final Clean Slate: $\\mathcal{R}_u^* = [i_1^*, \\dots, i_K^*]$", fontsize=9.0, weight='bold', ha='center', color="#0F172A", zorder=3)

    # =========================================================================
    # CLEAN INTER-STAGE GUTTER CONNECTORS (NO CLIPPING)
    # =========================================================================
    arrow_main = dict(arrowstyle="->,head_width=0.45,head_length=0.7", lw=2.8, color="#0F172A")
    arrow_graph = dict(arrowstyle="->,head_width=0.4,head_length=0.6", lw=2.4, color="#0284C7")

    # 1. Stage 1 -> Stage 2 (Raw Bipartite graph to MIRF)
    ax.annotate("", xy=(20.5, 73.0), xytext=(19.7, 73.0), arrowprops=arrow_main, zorder=6)

    # 2. Stage 2 -> Stage 3 (Normalized Laplacian graph pass-through Â)
    ax.annotate("", xy=(41.1, 75.0), xytext=(40.3, 75.0), arrowprops=arrow_graph, zorder=6)

    # 3. Stage 3 -> Stage 4 (Contrastive Embeddings E_U, E_I)
    ax.annotate("", xy=(60.7, 73.0), xytext=(59.9, 73.0), arrowprops=arrow_main, zorder=6)

    # 4. Stage 4 -> Stage 5 (Candidate Preference Slates s_head, s_tail)
    ax.annotate("", xy=(80.0, 73.0), xytext=(79.2, 73.0), arrowprops=arrow_main, zorder=6)

    # =========================================================================
    # DEDICATED RELIABILITY SIGNAL HIGHWAY (y: 13.0 to 16.8)
    # =========================================================================
    draw_rounded_card(ax, 20.5, 13.0, 78.3, 3.6, "#0F172A", "#F59E0B", lw=2.0, radius=0.5, zorder=3)
    ax.text(59.65, 14.8, "RELIABILITY SIGNAL HIGHWAY: Sample Trust Weights $\\mathbf{w_{ui} \\in [0, 1]}$ & Inverted Noise Operator $T^{\\dagger}_{\\lambda}$",
            weight='bold', fontsize=9.6, ha='center', va='center', color="#FDE047", zorder=4)

    # Downward drop from Stage 2 Fusion Hub to Highway
    ax.annotate("", xy=(30.4, 16.6), xytext=(30.4, 19.2),
                arrowprops=dict(arrowstyle="->,head_width=0.45,head_length=0.6", lw=2.6, color="#D97706"), zorder=5)
    draw_rounded_card(ax, 24.5, 17.0, 11.8, 1.8, "#D97706", "#B45309", radius=0.25, zorder=6)
    ax.text(30.4, 17.9, "BUS OUT: $w_{ui} \\in [0, 1]$", fontsize=7.6, weight='bold', ha='center', va='center', color="#FFFFFF", zorder=7)

    # Upward tap into Stage 3 (InfoNCE)
    ax.annotate("", xy=(50.5, 19.2), xytext=(50.5, 16.6),
                arrowprops=dict(arrowstyle="->,head_width=0.45,head_length=0.6", lw=2.6, color="#DB2777"), zorder=5)
    draw_rounded_card(ax, 44.5, 17.0, 12.0, 1.8, "#DB2777", "#9D174D", radius=0.25, zorder=6)
    ax.text(50.5, 17.9, "TAP: $w_{ui} \\to \\mathcal{L}_{\\mathrm{cl}}$", fontsize=7.6, weight='bold', ha='center', va='center', color="#FFFFFF", zorder=7)

    # Upward tap into Stage 4 (Risk Loss)
    ax.annotate("", xy=(69.9, 19.2), xytext=(69.9, 16.6),
                arrowprops=dict(arrowstyle="->,head_width=0.45,head_length=0.6", lw=2.6, color="#7C3AED"), zorder=5)
    draw_rounded_card(ax, 63.5, 17.0, 12.8, 1.8, "#7C3AED", "#5B21B6", radius=0.25, zorder=6)
    ax.text(69.9, 17.9, "TAP: $w_{ui}, T^{\\dagger}_{\\lambda} \\to \\mathcal{L}_{\\mathrm{risk}}$", fontsize=7.6, weight='bold', ha='center', va='center', color="#FFFFFF", zorder=7)

    # Upward tap into Stage 5 (Blueprint Pruning)
    ax.annotate("", xy=(89.4, 19.2), xytext=(89.4, 16.6),
                arrowprops=dict(arrowstyle="->,head_width=0.45,head_length=0.6", lw=2.6, color="#16A34A"), zorder=5)
    draw_rounded_card(ax, 83.2, 17.0, 12.4, 1.8, "#16A34A", "#15803D", radius=0.25, zorder=6)
    ax.text(89.4, 17.9, "TAP: $w_{ui} \\geq \\theta \\to \\mathcal{B}_u$", fontsize=7.6, weight='bold', ha='center', va='center', color="#FFFFFF", zorder=7)

    # =========================================================================
    # BOTTOM BENCHMARK VALIDATION & TELEMETRY CONSOLE (y: 1.2 to 12.2)
    # =========================================================================
    draw_rounded_card(ax, 1.2, 1.2, 97.6, 11.0, "#042F2E", "#0D9488", lw=2.0, radius=0.8, zorder=2)
    ax.text(50.0, 10.9, "ENTERPRISE TELEMETRY, MULTI-DOMAIN EMPIRICAL BENCHMARKS & OFFLINE A/B ECONOMIC IMPACT",
            weight='bold', fontsize=11.2, ha='center', va='center', color="#2DD4BF", zorder=3)

    metrics = [
        ("Adversarial Robustness", "ROC-AUC: 0.7931 ± 0.0038\nAttack AP: 0.8879 vs 0.2104\nAuthentic AP: 0.9989 (MovieLens)", "#14B8A6"),
        ("Recommendation Fidelity", "Recall@10: 0.2307 ± 0.0034\nnDCG@10: 0.1036 ± 0.0021\nStatistically Significant (p < 0.001)", "#38BDF8"),
        ("Popularity Calibration", "RMSE-PC: 0.1067 (p = 0.0166*)\nMRMC: 0.1128 (p = 0.0088*)\nSPAM STER@10 Drop: 88% → 72%", "#A78BFA"),
        ("Offline Economic Lift", "CTR Lift: +3.64% (p < 0.001)\nAnnual GMV Lift: +$534,491\n1,000-Sample Monte Carlo Sim", "#34D399"),
        ("Inference & Acceleration", "2D BLAS GEMM: 4.95 ms (42K edges)\nSpeedup: >500x vs Iterative\nLLM WAL Cache Hit Rate: 88.4%", "#FBBF24"),
    ]

    for idx, (m_title, m_desc, m_col) in enumerate(metrics):
        mx = 2.4 + idx * 19.3
        draw_rounded_card(ax, mx, 1.9, 18.5, 7.8, "#134E4A", m_col, lw=1.2, radius=0.4, zorder=3)
        ax.text(mx + 9.25, 8.4, m_title, weight='bold', fontsize=9.0, ha='center', color=m_col, zorder=4)
        ax.text(mx + 9.25, 4.8, m_desc, fontsize=7.6, weight='bold', ha='center', va='center', color="#E6FFFA", zorder=4)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig1_system_architecture.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, "fig1_system_architecture.png"), bbox_inches='tight')
    plt.close()
    print("Saved high-legibility, zero-collision 5-stage fig1_system_architecture.[pdf/png]")


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
             r"$\alpha_u = \frac{\sum_{\mathrm{tail}} w_{ui}}{\sum w_{ui} + \epsilon}$"
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
    roc_auc = 0.7931

    if len(fpr) > 1000:
        step = len(fpr) // 500
        fpr = fpr[::step]
        tpr = tpr[::step]

    ax1.plot(fpr, tpr, color='#1f77b4', lw=2.6, label=f'MIRF Filter (AUC = {roc_auc:.4f})')
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
    pr_auc = 0.9990

    if len(precision) > 1000:
        step = len(precision) // 500
        precision = precision[::step]
        recall = recall[::step]

    ax2.plot(recall, precision, color='#d62728', lw=2.6, label=f'MIRF Filter (AP = {pr_auc:.4f})')
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

    ml_super = [0.2307 / 0.30, 0.1036 / 0.15, 0.1430 / 0.25, 11.68 / 15.0, 0.7931]
    ml_vanilla = [0.2147 / 0.30, 0.1002 / 0.15, 0.1575 / 0.25, 11.72 / 15.0, 0.5000]

    amz_super = [0.2600 / 0.40, 0.1051 / 0.20, 0.5079 / 0.60, 6.47 / 8.0, 0.7867]
    amz_vanilla = [0.2967 / 0.40, 0.1326 / 0.20, 0.4526 / 0.60, 6.42 / 8.0, 0.5000]

    yelp_super = [0.2229 / 0.30, 0.1017 / 0.15, 0.4298 / 0.60, 6.77 / 8.5, 0.7961]
    yelp_vanilla = [0.2400 / 0.30, 0.1169 / 0.15, 0.3511 / 0.60, 6.56 / 8.5, 0.5000]

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

    plot_radar(ax1, ml_super, ml_vanilla, '(a) MovieLens-1M')
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
    Figure 8: 4-Panel High-Resolution Interactive Dashboard Demonstration
    Combines user-captured live telemetry screenshots:
    - (a) Tab 1: Multi-View MIRF Signal Fusion & Weight Distribution
    - (b) Tab 2: Noise Transition Matrix & Denoising Diagnostics
    - (c) Tab 3: Academic Benchmark & Head-to-Head Robustness Curves
    - (d) Tab 7: A/B Business Impact Simulator & GMV Scaling
    """
    img_t1_path = os.path.join(OUTPUT_DIR, "screenshot_tab1_multiview_fusion.png")
    img_t2_path = os.path.join(OUTPUT_DIR, "screenshot_tab2_noise_transition.png")
    img_t3_path = os.path.join(OUTPUT_DIR, "screenshot_tab3_academic_benchmark.png")
    img_t7_path = os.path.join(OUTPUT_DIR, "screenshot_tab7_ab_simulator.png")

    if not all(os.path.exists(p) for p in [img_t1_path, img_t2_path, img_t3_path, img_t7_path]):
        print("Skipping dashboard collage: one or more screenshot files missing.")
        return

    im1 = Image.open(img_t1_path)
    im2 = Image.open(img_t2_path)
    im3 = Image.open(img_t3_path)
    im7 = Image.open(img_t7_path)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15.0, 9.2))

    ax1.imshow(im1)
    ax1.set_title("(a) Multi-View MIRF Signal Fusion & Weight Distribution Telemetry", fontsize=13.0, weight='bold', pad=8)
    ax1.axis('off')

    ax2.imshow(im2)
    ax2.set_title("(b) Noise Transition Matrix Inversion & Denoising Diagnostics", fontsize=13.0, weight='bold', pad=8)
    ax2.axis('off')

    ax3.imshow(im3)
    ax3.set_title("(c) Head-to-Head Academic Benchmarks & Attack Budget Curves", fontsize=13.0, weight='bold', pad=8)
    ax3.axis('off')

    ax4.imshow(im7)
    ax4.set_title("(d) A/B Business Impact Simulator & GMV Revenue Scaling", fontsize=13.0, weight='bold', pad=8)
    ax4.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig8_dashboard_multipanel.png"), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, "fig8_dashboard_multipanel.pdf"), bbox_inches='tight')
    plt.close()
    print("Saved high-res fig8_dashboard_multipanel.[pdf/png] using user screenshots.")


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
