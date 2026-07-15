#!/usr/bin/env python3
"""Generate the H2L AUROC detector heatmap grid as a paper-ready PDF."""

from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "results"
DEFAULT_OUTPUT = ROOT / "paper" / "assets" / "heatmap_h2l_auroc"

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
GENERATOR_HINTS = [
    ("vicuna", "Vicuna"),
    ("llama", "Llama"),
    ("gemma", "Gemma"),
    ("deepseek", "DeepSeek"),
]


def parse_args(
    description: str = __doc__ or "",
    default_output: Path = DEFAULT_OUTPUT,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing detector result .npz files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Output path without an extension.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("pdf",),
        help="Output formats to write. Defaults to PDF only.",
    )
    return parser.parse_args()


def load_metrics(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    required = {
        "detector_name",
        "dataset_name",
        "generator_name",
        "target_regime",
        "auroc",
        "tpr_at_fpr",
    }

    for path in sorted(results_dir.rglob("*.npz")):
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
                "auroc": float(metadata["auroc"]),
                "tpr_at_fpr": float(metadata["tpr_at_fpr"]),
            }
        )

    if not rows:
        raise FileNotFoundError(f"No .npz result files found in {results_dir}")
    return pd.DataFrame(rows)


def preferred_order(values: list[str], preferred: list[str]) -> list[str]:
    present = list(dict.fromkeys(values))
    preferred_present = [value for value in preferred if value in present]
    extras = [value for value in present if value not in preferred_present]
    return preferred_present + sorted(extras, key=str.casefold)


def generator_sort_key(value: str) -> tuple[int, str]:
    lowered = str(value).casefold()
    for idx, (hint, _) in enumerate(GENERATOR_HINTS):
        if hint in lowered:
            return idx, lowered
    return len(GENERATOR_HINTS), lowered


def generator_label(value: str) -> str:
    lowered = str(value).casefold()
    for hint, label in GENERATOR_HINTS:
        if hint in lowered:
            return label
    return str(value)


def build_metric_frame(
    metrics: pd.DataFrame,
    metric: str,
    target_regime: str = "h2l",
) -> pd.DataFrame:
    selected = metrics.loc[
        metrics["regime"].eq(target_regime)
        & metrics["detector"].isin(DETECTOR_ORDER)
        & metrics["dataset"].isin(DATASET_ORDER),
        ["detector", "dataset", "generator", metric],
    ].copy()

    if selected.empty:
        raise ValueError(f"No rows matched target_regime={target_regime!r}")

    counts = (
        selected.groupby(["detector", "dataset"])["generator"]
        .nunique()
        .rename("n_generators")
    )
    if counts.nunique() != 1:
        raise ValueError(
            "Unequal generator coverage across detector × dataset blocks: "
            f"{counts.to_dict()}"
        )

    return selected


def save_figure(fig: plt.Figure, output: Path, formats: tuple[str, ...]) -> list[Path]:
    written: list[Path] = []
    for fmt in formats:
        target = output.with_suffix(f".{fmt}")
        target.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.03}
        if fmt.lower() == "png":
            save_kwargs["dpi"] = 600
        fig.savefig(target, format=fmt, **save_kwargs)
        written.append(target)
    return written


def plot_heatmap_grid(
    frame: pd.DataFrame,
    metric: str,
    colorbar_label: str,
    output: Path,
    formats: tuple[str, ...] = ("pdf",),
) -> tuple[list[Path], pd.DataFrame]:
    detectors = preferred_order(frame["detector"].dropna().tolist(), DETECTOR_ORDER)
    datasets = preferred_order(frame["dataset"].dropna().tolist(), DATASET_ORDER)
    generators = sorted(frame["generator"].dropna().unique(), key=generator_sort_key)

    sns.set_theme(style="white", context="paper", font="Times New Roman")
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 15,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    ncols = 2
    nrows = ceil(len(detectors) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.2, 9.4),
        squeeze=False,
        constrained_layout=True,
    )
    flat_axes = axes.ravel()
    visible_axes = []

    for panel_idx, (ax, detector) in enumerate(zip(flat_axes, detectors)):
        detector_frame = frame[frame["detector"] == detector]
        matrix = detector_frame.pivot_table(
            index="generator",
            columns="dataset",
            values=metric,
            aggfunc="mean",
            sort=False,
        ).reindex(index=generators, columns=datasets)
        display = matrix.rename(index=generator_label, columns=DATASET_LABELS)

        sns.heatmap(
            display,
            ax=ax,
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            annot=True,
            fmt=".3f",
            linewidths=0.5,
            linecolor="white",
            cbar=False,
            annot_kws={"fontsize": 8},
        )
        ax.set_title(DETECTOR_LABELS.get(detector, detector))
        ax.set_xlabel("Dataset" if panel_idx + ncols >= len(detectors) else "")
        ax.set_ylabel("Generator" if panel_idx % ncols == 0 else "")
        ax.tick_params(axis="x", rotation=0, length=0)
        ax.tick_params(axis="y", rotation=0, length=0)
        visible_axes.append(ax)

    for ax in flat_axes[len(detectors) :]:
        ax.set_visible(False)

    colorbar = fig.colorbar(
        visible_axes[0].collections[0],
        ax=visible_axes,
        location="right",
        shrink=0.82,
        pad=0.02,
    )
    colorbar.set_label(colorbar_label)

    aggregated = (
        frame.groupby(["detector", "generator", "dataset"], as_index=False)[metric]
        .mean()
        .sort_values(
            by=["detector", "generator", "dataset"],
            key=lambda column: column.map(
                {
                    **{name: idx for idx, name in enumerate(DETECTOR_ORDER)},
                    **{name: idx for idx, name in enumerate(DATASET_ORDER)},
                }
            )
            if column.name in {"detector", "dataset"}
            else column.map(
                {name: idx for idx, name in enumerate(generators)}
            ),
        )
    )

    written = save_figure(fig, output, formats=formats)
    plt.close(fig)
    return written, aggregated


def main() -> None:
    args = parse_args()
    metric_frame = build_metric_frame(load_metrics(args.results_dir), metric="auroc")
    written, aggregated = plot_heatmap_grid(
        metric_frame,
        metric="auroc",
        colorbar_label="AUROC",
        output=args.output,
        formats=tuple(args.formats),
    )
    print(aggregated.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
