"""
test_m4_adversarial_challenger.py
---------------------------------
Adversarial Stress Test Suite for Milestone M4 (Requirement R4)
Authored by Empirical Challenger teamwork_preview_challenger_m4_1.

Tests include:
1. Adversarial scanning for deprecated legacy strings:
   - Assert 15.72% does NOT appear in paper/ or build_report/
   - Assert 0.1572 does NOT appear as random baseline in paper/ or build_report/
   - Assert 0.9990 does NOT appear in paper/generate_figures.py or paper/paper.tex
2. Adversarial check for Table IV standard deviations:
   - Assert \\pm 0.0000 is NOT present in Table IV for Amazon or Yelp RRFN-LLM-SUPER
   - Assert Table IV ROC-AUC std devs match multiseed_benchmark.json (+-0.0208 Amazon, +-0.0232 Yelp)
3. Mathematical precision:
   - Verify Yelp lift calculation: 0.6835 / 0.1486 = 4.5996... approx 4.60x
   - Verify narrative alignment in paper.tex and build_report/
4. Monotonic ablation table integrity:
   - Verify Table V in paper.tex matches robust_super/results/run_15epoch_gpu/ablation_table.json:
     * Full: 0.2333, 0.1069, 0.1531, 0.7028
     * no_rrfn: 0.2467, 0.1120, 0.1505, 0.4416 (-37.2%)
     * no_llm: 0.2133, 0.0963, 0.1603, 0.6515 (-7.3%)
     * no_bomb: 0.2267, 0.0930, 0.1502, 0.3953 (-43.8%)
     * no_gcl: 0.2400, 0.1124, 0.1492, 0.7028 (-2.6%)
     * no_super: 0.1200, 0.0492, 0.0849, 0.7028 (-44.5%)
5. Publication figure artifacts:
   - Verify all 8 figures exist in both .pdf and .png format and are non-empty
   - Verify loss_history.json contains full 15 epochs across all 4 tracked series
"""

import os
import json
import re
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PAPER_DIR = os.path.join(PROJECT_ROOT, "paper")
BUILD_REPORT_DIR = os.path.join(PROJECT_ROOT, "build_report")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "robust_super", "results")


def test_adversarial_legacy_strings_paper():
    """Assert 15.72%, 0.1572, and 0.9990 do not appear in paper/ source files."""
    forbidden = ["15.72%", "0.1572", "0.9990"]
    for root, dirs, files in os.walk(PAPER_DIR):
        if "figures" in root or ".git" in root or "__pycache__" in root:
            continue
        for f in files:
            if f.endswith((".tex", ".py")):
                path = os.path.join(root, f)
                with open(path, "r", encoding="utf-8", errors="ignore") as fp:
                    content = fp.read()
                    for bad in forbidden:
                        assert bad not in content, f"Legacy forbidden string '{bad}' found in {path}"


def test_adversarial_legacy_strings_build_report():
    """Assert 15.72%, 0.1572, and 0.9990 do not appear in build_report/ source files."""
    forbidden = ["15.72%", "0.1572", "0.9990"]
    for root, dirs, files in os.walk(BUILD_REPORT_DIR):
        if "__pycache__" in root:
            continue
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                with open(path, "r", encoding="utf-8", errors="ignore") as fp:
                    content = fp.read()
                    for bad in forbidden:
                        assert bad not in content, f"Legacy forbidden string '{bad}' found in {path}"


def test_adversarial_yelp_lift_mathematics():
    """Verify Yelp attack detection lift formula: 0.6835 / 0.1486 = 4.5996... approx 4.60x."""
    yelp_precision = 0.6835
    yelp_prevalence = 0.1486
    computed_lift = yelp_precision / yelp_prevalence
    expected_lift = 4.599596231493944
    assert abs(computed_lift - expected_lift) < 1e-6
    assert round(computed_lift, 2) == 4.60
    assert f"{computed_lift:.1f}" == "4.6"

    # Verify paper.tex contains the reconciled lift
    paper_tex_path = os.path.join(PAPER_DIR, "paper.tex")
    with open(paper_tex_path, "r", encoding="utf-8") as f:
        tex = f.read()
    assert r"$4.6\times$ lift over the $0.1486$ random baseline" in tex
    assert r"$3.9$--$4.6\times$" in tex or r"3.9--4.6\times" in tex


def test_adversarial_table_iv_standard_deviations():
    """Verify Table IV ROC-AUC standard deviations match multiseed_benchmark.json and are NOT 0.0000."""
    paper_tex_path = os.path.join(PAPER_DIR, "paper.tex")
    with open(paper_tex_path, "r", encoding="utf-8") as f:
        tex = f.read()

    benchmark_json_path = os.path.join(RESULTS_DIR, "multiseed_benchmark.json")
    with open(benchmark_json_path, "r", encoding="utf-8") as f:
        bench = json.load(f)

    amz_std = bench["amazon"]["summary_stats"]["robust"]["Denoising-ROC-AUC"]["std"]
    yelp_std = bench["yelp"]["summary_stats"]["robust"]["Denoising-ROC-AUC"]["std"]

    assert round(amz_std, 4) == 0.0208, f"Amazon std dev {amz_std} did not round to 0.0208"
    assert round(yelp_std, 4) == 0.0232, f"Yelp std dev {yelp_std} did not round to 0.0232"

    # Locate Table IV by label
    assert r"\label{tab:multidomain_benchmark}" in tex, "Table IV label not found in paper.tex"
    table_start = tex.find(r"\label{tab:multidomain_benchmark}")
    table_end = tex.find(r"\end{table*}", table_start)
    assert table_end != -1, "Table IV end not found"
    table_text = tex[table_start:table_end]

    # Amazon RRFN-LLM-SUPER row
    amz_start = table_text.find(r"Amazon (SimGCL)")
    yelp_start = table_text.find(r"Yelp (SGL)")
    assert amz_start != -1 and yelp_start != -1

    amz_block = table_text[amz_start:yelp_start]
    yelp_block = table_text[yelp_start:]

    # Amazon RRFN-LLM-SUPER row line
    amz_ours_line = [l for l in amz_block.splitlines() if "RRFN-LLM-SUPER" in l]
    assert len(amz_ours_line) == 1, f"Expected 1 Amazon RRFN-LLM-SUPER line, found {len(amz_ours_line)}"
    amz_row = amz_ours_line[0]
    assert r"\pm 0.0000" not in amz_row, f"Amazon RRFN-LLM-SUPER row still has \\pm 0.0000: {amz_row}"
    assert r"0.7867 \pm 0.0208" in amz_row, f"Amazon row missing 0.7867 \\pm 0.0208: {amz_row}"

    # Yelp RRFN-LLM-SUPER row line
    yelp_ours_line = [l for l in yelp_block.splitlines() if "RRFN-LLM-SUPER" in l]
    assert len(yelp_ours_line) == 1, f"Expected 1 Yelp RRFN-LLM-SUPER line, found {len(yelp_ours_line)}"
    yelp_row = yelp_ours_line[0]
    assert r"\pm 0.0000" not in yelp_row, f"Yelp RRFN-LLM-SUPER row still has \\pm 0.0000: {yelp_row}"
    assert r"0.7961 \pm 0.0232" in yelp_row, f"Yelp row missing 0.7961 \\pm 0.0232: {yelp_row}"


def test_adversarial_table_v_ablation_entries():
    """Verify Table V ablation entries match ablation_table.json to 4 decimal places."""
    ablation_json_path = os.path.join(RESULTS_DIR, "run_15epoch_gpu", "ablation_table.json")
    assert os.path.exists(ablation_json_path), f"Missing {ablation_json_path}"
    with open(ablation_json_path, "r", encoding="utf-8") as f:
        abl_data = json.load(f)

    paper_tex_path = os.path.join(PAPER_DIR, "paper.tex")
    with open(paper_tex_path, "r", encoding="utf-8") as f:
        tex = f.read()

    # Expected values
    expected = {
        "Full": {"recall": 0.2333, "ndcg": 0.1069, "ltc": 0.1531, "roc": 0.7028, "drop": 0.0},
        "no_rrfn": {"recall": 0.2467, "ndcg": 0.1120, "ltc": 0.1505, "roc": 0.4416, "drop": -37.2},
        "no_llm": {"recall": 0.2133, "ndcg": 0.0963, "ltc": 0.1603, "roc": 0.6515, "drop": -7.3},
        "no_bomb": {"recall": 0.2267, "ndcg": 0.0930, "ltc": 0.1502, "roc": 0.3953, "drop": -43.8},
        "no_gcl": {"recall": 0.2400, "ndcg": 0.1124, "ltc": 0.1492, "roc": 0.7028, "drop": -2.6},
        "no_super": {"recall": 0.1200, "ndcg": 0.0492, "ltc": 0.0849, "roc": 0.7028, "drop": -44.5},
    }

    # Verify JSON matches expected
    keys = ["Full", "no_rrfn", "no_llm", "no_bomb", "no_gcl", "no_super"]
    for row, k in zip(abl_data, keys):
        exp = expected[k]
        assert round(row["Recall@10"], 4) == exp["recall"], f"{k} Recall@10 mismatch"
        assert round(row["nDCG@10"], 4) == exp["ndcg"], f"{k} nDCG@10 mismatch"
        assert round(row["LTC@10"], 4) == exp["ltc"], f"{k} LTC@10 mismatch"
        assert round(row["Denoising_ROC"], 4) == exp["roc"], f"{k} ROC mismatch"
        assert row["Delta_Drop_Pct"] == exp["drop"], f"{k} Delta drop mismatch"

    # Verify Table V in paper.tex contains these exact numerical strings
    assert r"\label{tab:ablation}" in tex, "Table V label not found in paper.tex"
    table_v_start = tex.find(r"\label{tab:ablation}")
    table_v_end = tex.find(r"\end{table}", table_v_start)
    table_v_text = tex[table_v_start:table_v_end]

    assert "0.2333" in table_v_text and "0.1069" in table_v_text and "0.1531" in table_v_text and "0.7028" in table_v_text
    assert "0.2467" in table_v_text and "0.1120" in table_v_text and "0.1505" in table_v_text and "0.4416" in table_v_text
    assert "0.2133" in table_v_text and "0.0963" in table_v_text and "0.1603" in table_v_text and "0.6515" in table_v_text
    assert "0.2267" in table_v_text and "0.0930" in table_v_text and "0.1502" in table_v_text and "0.3953" in table_v_text
    assert "0.2400" in table_v_text and "0.1124" in table_v_text and "0.1492" in table_v_text
    assert "0.1200" in table_v_text and "0.0492" in table_v_text and "0.0849" in table_v_text


def test_adversarial_figure_artifacts():
    """Verify all 8 figures exist in both .pdf and .png formats with size > 1KB."""
    figure_names = [
        "fig1_system_architecture",
        "fig2_mirf_pipeline",
        "fig3_roc_pr_curves",
        "fig4_omega_sensitivity",
        "fig5_multidomain_radar",
        "fig6_loss_curves",
        "fig7_super_blueprint_flow",
        "fig8_dashboard_multipanel",
    ]
    figures_dir = os.path.join(PAPER_DIR, "figures")
    for name in figure_names:
        for ext in [".pdf", ".png"]:
            path = os.path.join(figures_dir, f"{name}{ext}")
            assert os.path.exists(path), f"Missing figure file: {path}"
            size = os.path.getsize(path)
            assert size > 1000, f"Figure file suspiciously small ({size} bytes): {path}"


def test_adversarial_loss_history_15_epochs():
    """Verify loss_history.json contains exactly 15 epochs across all series."""
    loss_path = os.path.join(RESULTS_DIR, "run_15epoch_gpu", "loss_history.json")
    assert os.path.exists(loss_path), f"Missing {loss_path}"
    with open(loss_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for key in ["M_pop_train_loss", "M_tail_train_loss", "M_pop_val_recall", "M_tail_val_recall"]:
        assert key in data, f"Missing key {key} in loss_history.json"
        assert len(data[key]) == 15, f"Series {key} does not have length 15 (got {len(data[key])})"
