#!/usr/bin/env python3
"""Generate Figure 3: detector TPR@1%FPR across generation regimes."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.evaluation import plot_detection_gap_barplot  # noqa: E402


RESULTS_DIR = ROOT / "results"
OUTPUT = ROOT / "paper" / "input" / "figure3_detection_gap"
DETECTOR_ORDER = [
    "BERT-Defense",
    "RoBERTa-Defense",
    "FastDetectGPT",
    "binoculars-falcon-7b",
    "radar",
]


def main() -> None:
    _, _, summary = plot_detection_gap_barplot(
        RESULTS_DIR,
        detector_names=DETECTOR_ORDER,
        regimes=("free_llm", "llm2l", "h2l"),
        n_bootstrap=5000,
        random_seed=42,
        alpha=0.05,
        # title="Detection performance across generation regimes",
        title="",
        output_path=OUTPUT,
        formats=("pdf", "png"),
    )
    columns = [
        "detector",
        "target_regime",
        "mean_tpr_at_fpr",
        "tpr_ci_low",
        "tpr_ci_high",
        "n_blocks",
    ]
    print(summary[columns].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Wrote {OUTPUT.with_suffix('.pdf')}")
    print(f"Wrote {OUTPUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
