import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

class PublicationPlotter:
    """
    Generates publication-quality figures (PDF + PNG) for research paper insertion.
    Uses IEEE / ACM styling (high DPI, professional palettes, serif fonts).
    """
    def __init__(self, output_dir: str = "experiments/figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._setup_style()

    def _setup_style(self):
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight"
        })
        sns.set_theme(style="whitegrid", palette="deep")

    def plot_attack_degradation(
        self,
        noise_rates: List[float],
        vanilla_ndcg: List[float],
        robust_ndcg: List[float],
        vanilla_rmse: List[float],
        robust_rmse: List[float],
        attack_name: str = "Bandwagon Attack",
        filename: str = "fig1_attack_curves"
    ):
        """Figure 1: nDCG@10 and RMSE-PC vs Attack Noise Rate."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # (a) Ranking Accuracy (nDCG@10)
        axes[0].plot(noise_rates, vanilla_ndcg, "o--", color="#d9534f", label="Vanilla SUPER (Uncorrected)", linewidth=2)
        axes[0].plot(noise_rates, robust_ndcg, "s-", color="#0275d8", label="RRFN-LLM-SUPER (Ours)", linewidth=2)
        axes[0].set_xlabel("Attack Budget (Noise Rate $\\rho$)")
        axes[0].set_ylabel("nDCG@10 (Higher is Better)")
        axes[0].set_title(f"(a) Ranking Accuracy under {attack_name}")
        axes[0].legend(frameon=True)

        # (b) Calibration Error (RMSE-PC)
        axes[1].plot(noise_rates, vanilla_rmse, "o--", color="#d9534f", label="Vanilla SUPER (Uncorrected)", linewidth=2)
        axes[1].plot(noise_rates, robust_rmse, "s-", color="#5cb85c", label="RRFN-LLM-SUPER (Ours)", linewidth=2)
        axes[1].set_xlabel("Attack Budget (Noise Rate $\\rho$)")
        axes[1].set_ylabel("RMSE-PC (Lower is Better)")
        axes[1].set_title(f"(b) Popularity Calibration under {attack_name}")
        axes[1].legend(frameon=True)

        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, f"{filename}.pdf"))
        fig.savefig(os.path.join(self.output_dir, f"{filename}.png"))
        plt.close(fig)

    def plot_transition_matrix(
        self,
        T_hat: np.ndarray,
        T_final: np.ndarray,
        filename: str = "fig2_transition_heatmap"
    ):
        """Figure 2: Noise Transition Matrix Heatmaps (Estimated vs Corrected)."""
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        classes = ["1★", "2★", "3★", "4★", "5★"]

        sns.heatmap(T_hat, ax=axes[0], annot=True, fmt=".2f", cmap="Blues", cbar=True, xticklabels=classes, yticklabels=classes)
        axes[0].set_title("(a) Empirical Estimate $\\hat{T}$ (Anchor Points)")
        axes[0].set_xlabel("Observed Rating $\\tilde{Y}$")
        axes[0].set_ylabel("True Clean Rating $Y^*$")

        sns.heatmap(T_final, ax=axes[1], annot=True, fmt=".2f", cmap="Blues", cbar=True, xticklabels=classes, yticklabels=classes)
        axes[1].set_title("(b) Refined Transition Matrix $T_{final} = \\text{Softmax}(\\hat{T} + \\Delta T)$")
        axes[1].set_xlabel("Observed Rating $\\tilde{Y}$")
        axes[1].set_ylabel("True Clean Rating $Y^*$")

        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, f"{filename}.pdf"))
        fig.savefig(os.path.join(self.output_dir, f"{filename}.png"))
        plt.close(fig)

    def plot_reliability_distribution(
        self,
        weights: Dict[tuple, float],
        ground_truth: Dict[tuple, int],
        filename: str = "fig3_weight_distributions"
    ):
        """Figure 3: Probability density of fused reliability weights for genuine vs injected interactions."""
        fig, ax = plt.subplots(figsize=(7, 4.5))

        genuine_w = [w for k, w in weights.items() if ground_truth.get(k, 1) == 1]
        injected_w = [w for k, w in weights.items() if ground_truth.get(k, 1) == 0]

        sns.kdeplot(genuine_w, ax=ax, label="Genuine Interactions (Label=1)", color="#0275d8", fill=True, alpha=0.4, linewidth=2)
        if injected_w:
            sns.kdeplot(injected_w, ax=ax, label="Injected / Corrupted (Label=0)", color="#d9534f", fill=True, alpha=0.4, linewidth=2)

        ax.axvline(0.50, color="gray", linestyle="--", label="Decision Threshold ($w=0.5$)")
        ax.set_xlabel("Fused Reliability Weight $w_{ui}$")
        ax.set_ylabel("Density")
        ax.set_title("Multi-View Reliability Weight Distribution ($w_{ui}$)")
        ax.legend(loc="upper center", frameon=True)

        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, f"{filename}.pdf"))
        fig.savefig(os.path.join(self.output_dir, f"{filename}.png"))
        plt.close(fig)

    def plot_ablation_comparison(
        self,
        variant_names: List[str],
        gkpi_scores: List[float],
        filename: str = "fig6_ablation_gkpi"
    ):
        """Figure 6: GKPI across 7 Ablation Variants."""
        fig, ax = plt.subplots(figsize=(9, 4.5))
        colors = ["#0275d8" if "full" in v.lower() else "#6c757d" for v in variant_names]

        bars = ax.barh(variant_names, gkpi_scores, color=colors, height=0.6, edgecolor="black", alpha=0.85)
        ax.set_xlabel("General Key Performance Indicator (GKPI)")
        ax.set_title("Ablation Study: Contribution of Individual Components (Bandwagon 10%)")
        ax.set_xlim(0, max(gkpi_scores) * 1.25)

        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.005, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontsize=10, fontweight="bold")

        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, f"{filename}.pdf"))
        fig.savefig(os.path.join(self.output_dir, f"{filename}.png"))
        plt.close(fig)
