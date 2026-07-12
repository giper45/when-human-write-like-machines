#!/usr/bin/env python3
"""Generate Figure 2: mean LLM2L delta TPR@1%FPR by detector and dataset."""

from pathlib import Path

from generate_figure1_delta_tpr_h2l import (
    ROOT,
    build_delta_matrix,
    load_metrics,
    plot_heatmap,
)


RESULTS_DIR = ROOT / "results"
OUTPUT = ROOT / "paper" / "input" / "figure2_delta_tpr_llm2l"


def main() -> None:
    matrix = build_delta_matrix(load_metrics(RESULTS_DIR), target_regime="llm2l")
    plot_heatmap(matrix, OUTPUT, target_label="LLM2L")
    print(matrix.to_string(float_format=lambda value: f"{value:.3f}"))
    print(f"Wrote {OUTPUT.with_suffix('.pdf')}")
    print(f"Wrote {OUTPUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
