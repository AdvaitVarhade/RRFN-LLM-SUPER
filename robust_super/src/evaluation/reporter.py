import os
import pandas as pd
from typing import Dict, List, Any, Optional

class ExperimentReporter:
    """
    Collects, aggregates, formats, and exports experiment benchmarks to CSV and LaTeX.
    """
    def __init__(self, output_dir: str = "experiments/results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.records: List[Dict[str, Any]] = []

    def record(
        self,
        method: str,
        backbone: str,
        attack_type: str,
        noise_rate: float,
        metrics: Dict[str, float],
        robustness: Optional[Dict[str, float]] = None
    ):
        entry = {
            "Method": method,
            "Backbone": backbone,
            "Attack": attack_type,
            "NoiseRate": noise_rate,
        }
        entry.update(metrics)
        if robustness:
            entry.update(robustness)
        self.records.append(entry)

    def export_csv(self, filename: str = "results_summary.csv") -> str:
        df = pd.DataFrame(self.records)
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, index=False)
        return path

    def export_latex(self, filename: str = "results_table.tex") -> str:
        df = pd.DataFrame(self.records)
        path = os.path.join(self.output_dir, filename)
        
        # Format floating point numbers to 4 decimals
        float_cols = df.select_dtypes(include=["float"]).columns
        for col in float_cols:
            df[col] = df[col].apply(lambda x: f"{x:.4f}")

        latex_str = df.to_latex(index=False, escape=False)
        with open(path, "w", encoding="utf-8") as f:
            f.write(latex_str)
        return path

    def print_summary(self):
        df = pd.DataFrame(self.records)
        if not df.empty:
            print("\n" + "="*80)
            print("EXPERIMENT BENCHMARK SUMMARY")
            print("="*80)
            print(df.to_string(index=False))
            print("="*80 + "\n")
