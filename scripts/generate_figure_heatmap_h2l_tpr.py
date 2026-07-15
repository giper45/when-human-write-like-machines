#!/usr/bin/env python3
"""Generate the H2L TPR@1%FPR detector heatmap grid as a paper-ready PDF."""

from __future__ import annotations

from generate_figure_heatmap_h2l_auroc import (
    ROOT,
    build_metric_frame,
    load_metrics,
    parse_args,
    plot_heatmap_grid,
)


DEFAULT_OUTPUT = ROOT / "paper" / "assets" / "heatmap_h2l_tpr_at_fpr"


def main() -> None:
    args = parse_args(description=__doc__ or "", default_output=DEFAULT_OUTPUT)
    metric_frame = build_metric_frame(load_metrics(args.results_dir), metric="tpr_at_fpr")
    written, aggregated = plot_heatmap_grid(
        metric_frame,
        metric="tpr_at_fpr",
        colorbar_label="TPR@1% FPR",
        output=args.output,
        formats=tuple(args.formats),
    )
    print(aggregated.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
