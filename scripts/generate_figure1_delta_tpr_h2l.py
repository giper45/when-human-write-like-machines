#!/usr/bin/env python3
"""Generate Figure 1: mean H2L delta TPR@1%FPR by detector and dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "results"
DEFAULT_OUTPUT = ROOT / "paper" / "input" / "figure1_delta_tpr_h2l"

DETECTOR_ORDER = [
    "BERT-Defense",
    "RoBERTa-Defense",
    "FastDetectGPT",
    "binoculars-falcon-7b",
    "radar",
]
DETECTOR_LABELS = {
    "binoculars-falcon-7b": "Binoculars-falcon-7b",
    "radar": "RADAR",
}
DATASET_ORDER = ["xsum", "wp", "owt"]
DATASET_LABELS = {
    "xsum": "XSum",
    "wp": "WritingPrompts",
    "owt": "OpenWebText",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing detector result .npz files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path without an extension (both PDF and PNG are written).",
    )
    return parser.parse_args()


def load_metrics(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    required = {
        "detector_name",
        "dataset_name",
        "generator_name",
        "target_regime",
        "tpr_at_fpr",
    }

    for path in sorted(results_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as result:
            if "metadata" not in result:
                raise ValueError(f"Missing metadata in {path}")
            metadata = json.loads(result["metadata"].item())

        missing = required - metadata.keys()
        if missing:
            raise ValueError(f"Missing {sorted(missing)} in {path}")
        rows.append(
            {
                "detector": metadata["detector_name"],
                "dataset": metadata["dataset_name"],
                "generator": metadata["generator_name"],
                "regime": metadata["target_regime"],
                "tpr_at_fpr": float(metadata["tpr_at_fpr"]),
            }
        )

    if not rows:
        raise FileNotFoundError(f"No .npz result files found in {results_dir}")
    return pd.DataFrame(rows)


def build_delta_matrix(
    metrics: pd.DataFrame, target_regime: str = "h2l"
) -> pd.DataFrame:
    selected = metrics[
        metrics["detector"].isin(DETECTOR_ORDER)
        & metrics["dataset"].isin(DATASET_ORDER)
        & metrics["regime"].isin(["free_llm", target_regime])
    ]

    key = ["detector", "dataset", "generator", "regime"]
    duplicates = selected.duplicated(key, keep=False)
    if duplicates.any():
        duplicated_keys = selected.loc[duplicates, key].to_dict("records")
        raise ValueError(f"Duplicate result blocks: {duplicated_keys}")

    paired = selected.pivot(
        index=["detector", "dataset", "generator"],
        columns="regime",
        values="tpr_at_fpr",
    )
    required_regimes = ["free_llm", target_regime]
    if not set(required_regimes).issubset(paired.columns):
        raise ValueError(f"Both free_llm and {target_regime} results are required")
    incomplete = paired[required_regimes].isna().any(axis=1)
    if incomplete.any():
        raise ValueError(f"Unpaired result blocks: {list(paired.index[incomplete])}")

    paired["delta"] = paired[target_regime] - paired["free_llm"]
    counts = paired.groupby(["detector", "dataset"])["delta"].count()
    if counts.nunique() != 1:
        raise ValueError(f"Unequal generator counts across cells: {counts.to_dict()}")

    matrix = paired.groupby(["detector", "dataset"])["delta"].mean().unstack()
    matrix = matrix.reindex(index=DETECTOR_ORDER, columns=DATASET_ORDER)
    if matrix.isna().any().any():
        raise ValueError("The detector-by-dataset matrix contains missing cells")
    return matrix


def plot_heatmap(
    matrix: pd.DataFrame, output: Path, target_label: str = "H2L"
) -> None:
    display = matrix.rename(index=DETECTOR_LABELS, columns=DATASET_LABELS)
    sns.set_theme(style="white", context="paper", font="DejaVu Sans")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.15))
    sns.heatmap(
        display,
        ax=ax,
        cmap=sns.diverging_palette(25, 245, s=85, l=48, as_cmap=True),
        center=0,
        vmin=-0.9,
        vmax=0.9,
        annot=True,
        fmt=".3f",
        linewidths=0.8,
        linecolor="white",
        cbar_kws={
            "label": (
                rf"Mean $\Delta$TPR@1%FPR ({target_label} $-$ FREE-LLM)"
            ),
            "shrink": 0.9,
            "pad": 0.025,
        },
        annot_kws={"fontsize": 9},
    )
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Detector")
    ax.tick_params(axis="x", rotation=0, length=0)
    ax.tick_params(axis="y", rotation=0, length=0)
    fig.tight_layout(pad=0.3)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    matrix = build_delta_matrix(load_metrics(args.results_dir))
    plot_heatmap(matrix, args.output)
    print(matrix.to_string(float_format=lambda value: f"{value:.3f}"))
    print(f"Wrote {args.output.with_suffix('.pdf')}")
    print(f"Wrote {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
