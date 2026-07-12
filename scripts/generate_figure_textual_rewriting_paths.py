#!/usr/bin/env python3
"""Generate the grouped bar plot comparing H2L and LLM2L text changes."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.evaluation import ResultsReportGraphBuilder  # noqa: E402


DEFAULT_INPUT = (
    ROOT / "results-aligned" / "text_analysis" / "h2l_vs_llm2l_text_metrics.csv"
)
DEFAULT_OUTPUT = ROOT / "paper" / "input" / "figure_textual_rewriting_paths"
METRIC_ORDER = ["Word ratio", "NED token", "Jaccard", "Semantic similarity"]
REGIME_ORDER = ["H2L", "LLM2L"]
PATH_LABELS = {
    "H2L": "HUMAN → H2L",
    "LLM2L": "FREE-LLM → LLM2L",
}
ESTIMATE_PATTERN = re.compile(
    r"^\s*(?P<mean>-?\d+(?:\.\d+)?)\s+"
    r"\[(?P<low>-?\d+(?:\.\d+)?),\s*(?P<high>-?\d+(?:\.\d+)?)\]\s*$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="CSV written by text_analysis.ipynb.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path without an extension (PDF and PNG are written).",
    )
    return parser.parse_args()


def load_text_metric_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run text_analysis.ipynb before generating the figure."
        )

    summary = pd.read_csv(path, index_col="Regime")
    missing_regimes = set(REGIME_ORDER) - set(summary.index)
    missing_metrics = set(METRIC_ORDER) - set(summary.columns)
    if missing_regimes or missing_metrics:
        raise ValueError(
            "Text metric summary has an unexpected schema: "
            f"missing regimes={sorted(missing_regimes)}, "
            f"missing metrics={sorted(missing_metrics)}"
        )

    rows: list[dict[str, object]] = []
    for regime in REGIME_ORDER:
        for metric in METRIC_ORDER:
            estimate = str(summary.loc[regime, metric])
            match = ESTIMATE_PATTERN.fullmatch(estimate)
            if match is None:
                raise ValueError(f"Invalid estimate for {regime}/{metric}: {estimate!r}")
            rows.append(
                {
                    "metric": metric,
                    "path": PATH_LABELS[regime],
                    "mean": float(match.group("mean")),
                    "ci_low": float(match.group("low")),
                    "ci_high": float(match.group("high")),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    summary = load_text_metric_summary(args.input)
    builder = ResultsReportGraphBuilder(formats=("pdf", "png"))
    builder.plot_grouped_barplot(
        data=summary,
        x="metric",
        y="mean",
        hue="path",
        output_path=args.output,
        title=None,
        ylabel="Mean metric value",
        xlabel=None,
        figsize=(7.1, 2.75),
        order=METRIC_ORDER,
        hue_order=[PATH_LABELS[regime] for regime in REGIME_ORDER],
        ci_low="ci_low",
        ci_high="ci_high",
        hatches=("", "///"),
        formats=("pdf", "png"),
        ylim=(0.0, 1.08),
        x_rotation=0,
        legend_title=None,
        legend_ncols=2,
        value_labels=True,
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Wrote {args.output.with_suffix('.pdf')}")
    print(f"Wrote {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
