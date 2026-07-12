from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


OKABE_ITO = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#F0E442",
    "#000000",
)
DEFAULT_HATCHES = ("", "///", "\\\\\\", "xx", "--", "++", "..", "oo", "**")
DEFAULT_LINESTYLES = ("-", "--", "-.", ":")
DEFAULT_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
VECTOR_FORMATS = {"pdf", "eps", "svg", "ps"}
RASTER_FORMATS = {"png", "tif", "tiff", "jpg", "jpeg"}


@dataclass
class ResultsReportGraphBuilder:
    """
    Build publication-ready result figures from detector evaluation CSV files.

    Plotting imports are lazy so metric/table code can run on machines where the
    optional figure dependencies have not been installed yet.
    """

    results_dir: Path | str = "results"
    output_dir: Path | str = "results/figures"
    formats: Sequence[str] = ("pdf",)
    dpi: int = 600
    style_applied: bool = field(default=False, init=False)

    def __post_init__(self):
        self.results_dir = Path(self.results_dir)
        self.output_dir = Path(self.output_dir)
        self.formats = tuple(str(fmt).lower().lstrip(".") for fmt in self.formats)

    @staticmethod
    def _require_plotting():
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError as exc:
            raise ImportError(
                "ResultsReportGraphBuilder requires matplotlib and seaborn. "
                "Install the project plotting dependencies before generating figures."
            ) from exc
        return plt, sns

    @staticmethod
    def set_paper_style() -> None:
        """Apply global Matplotlib/Seaborn settings for paper-ready figures."""
        plt, sns = ResultsReportGraphBuilder._require_plotting()
        sns.set_theme(style="whitegrid", context="paper", font="DejaVu Sans")
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 8,
                "axes.labelsize": 8,
                "axes.titlesize": 9,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "legend.fontsize": 7,
                "axes.linewidth": 0.8,
                "lines.linewidth": 1.5,
                "lines.markersize": 4,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "savefig.bbox": "tight",
                "savefig.pad_inches": 0.02,
                "grid.linewidth": 0.4,
                "grid.alpha": 0.35,
            }
        )

    def ensure_style(self) -> None:
        if not self.style_applied:
            self.set_paper_style()
            self.style_applied = True

    @staticmethod
    def get_colorblind_palette(n: int):
        """Return a colorblind-safe palette with n colors."""
        _, sns = ResultsReportGraphBuilder._require_plotting()
        if n <= len(OKABE_ITO):
            return list(OKABE_ITO[:n])
        return sns.color_palette("colorblind", n_colors=n)

    @staticmethod
    def format_axis(ax, xlabel=None, ylabel=None, title=None, grid_axis="y") -> None:
        """Apply consistent labels and axis formatting."""
        _, sns = ResultsReportGraphBuilder._require_plotting()
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        if grid_axis:
            ax.grid(True, axis=grid_axis, color="0.85", linewidth=0.4)
        sns.despine(ax=ax)

    @staticmethod
    def apply_bar_hatches(ax, hatches=None, hue_levels=None) -> None:
        """Apply hatch patterns to bar containers for grayscale readability."""
        hatches = tuple(hatches or DEFAULT_HATCHES)
        n_hue = len(hue_levels) if hue_levels is not None else None
        containers = [container for container in ax.containers if hasattr(container, "patches")]

        # Hatches make hue groups distinguishable when printed in grayscale.
        if n_hue and len(containers) >= n_hue:
            for hue_idx, container in enumerate(containers[:n_hue]):
                for patch in container.patches:
                    patch.set_hatch(hatches[hue_idx % len(hatches)])
                    patch.set_edgecolor("0.15")
                    patch.set_linewidth(0.5)
            return

        for patch_idx, patch in enumerate(ax.patches):
            patch.set_hatch(hatches[patch_idx % len(hatches)])
            patch.set_edgecolor("0.15")
            patch.set_linewidth(0.5)

    def save_figure(self, fig, output_path, formats=None, dpi=None) -> list[Path]:
        """
        Save a figure in publication-ready formats.

        Vector formats are preferred because they preserve text and lines cleanly
        through journal production workflows.
        """
        output_path = Path(output_path)
        formats = tuple(str(fmt).lower().lstrip(".") for fmt in (formats or self.formats))
        dpi = int(dpi or self.dpi)
        saved_paths = []

        for fmt in formats:
            target = output_path.with_suffix(f".{fmt}")
            target.parent.mkdir(parents=True, exist_ok=True)
            save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
            if fmt in RASTER_FORMATS:
                save_kwargs["dpi"] = dpi
            fig.savefig(target, format=fmt, **save_kwargs)
            saved_paths.append(target)

        return saved_paths

    @staticmethod
    def _as_frame(data) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data.copy()
        return pd.DataFrame(data)

    @staticmethod
    def _matrix_from_data(data, index=None, columns=None, values=None) -> pd.DataFrame:
        frame = ResultsReportGraphBuilder._as_frame(data)
        if values is None:
            return frame
        required = {name for name in (index, columns, values) if name is not None}
        if len(required) != 3:
            raise ValueError("Heatmap pivot requires index, columns, and values.")
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing heatmap columns: {sorted(missing)}")
        return frame.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")

    @staticmethod
    def _resolve_ci_columns(data: pd.DataFrame, y: str, ci_low=None, ci_high=None):
        low_candidates = [ci_low, f"{y}_ci_low", f"{y}_lower", f"{y}_low", "ci_low", "lower"]
        high_candidates = [ci_high, f"{y}_ci_high", f"{y}_upper", f"{y}_high", "ci_high", "upper"]
        low = next((col for col in low_candidates if col and col in data.columns), None)
        high = next((col for col in high_candidates if col and col in data.columns), None)
        return low, high

    @staticmethod
    def _ordered_unique(values: Iterable) -> list:
        return list(pd.Series(values).dropna().drop_duplicates())

    @staticmethod
    def _delta_cmap(cmap):
        _, sns = ResultsReportGraphBuilder._require_plotting()
        if isinstance(cmap, str) and cmap in {"vlag", "icefire"}:
            return sns.color_palette(cmap, as_cmap=True)
        return cmap

    def plot_delta_heatmap(
        self,
        data,
        output_path,
        title,
        cbar_label,
        index="detector",
        columns="dataset",
        values=None,
        figsize=(7.1, 3.2),
        cmap="vlag",
        center=0.0,
        fmt=".2f",
        annot=True,
        formats=None,
    ):
        """
        Plot a centered delta heatmap and save it.

        Delta heatmaps are centered at zero so gains and losses have symmetric
        visual weight. Numeric annotations keep the plot interpretable without
        relying on color alone.
        """
        plt, sns = self._require_plotting()
        self.ensure_style()
        matrix = self._matrix_from_data(data, index=index, columns=columns, values=values)

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            matrix,
            ax=ax,
            cmap=self._delta_cmap(cmap),
            center=center,
            annot=annot,
            fmt=fmt,
            linewidths=0.4,
            linecolor="0.85",
            cbar_kws={"label": cbar_label},
            annot_kws={"fontsize": 7},
        )
        self.format_axis(ax, xlabel=columns.title(), ylabel=index.title(), title=title, grid_axis=None)
        ax.tick_params(axis="x", rotation=35)
        ax.tick_params(axis="y", rotation=0)
        fig.tight_layout(pad=0.2)
        self.save_figure(fig, output_path, formats=formats)
        return fig, ax

    def load_npz_metrics(self) -> pd.DataFrame:
        """Load metric values and experiment dimensions from result metadata."""
        rows = []
        invalid_files = []
        required = {"dataset_name", "generator_name", "auroc", "tpr_at_fpr"}

        for path in sorted(self.results_dir.rglob("*.npz")):
            try:
                with np.load(path, allow_pickle=False) as result:
                    if "metadata" not in result:
                        raise KeyError("metadata")
                    metadata = json.loads(result["metadata"].item())

                missing = required - metadata.keys()
                if missing:
                    raise KeyError(", ".join(sorted(missing)))

                rows.append(
                    {
                        "dataset": metadata["dataset_name"],
                        "generator": metadata["generator_name"],
                        "detector": metadata.get("detector_name"),
                        "target_regime": metadata.get(
                            "target_regime", metadata.get("machine_postfix")
                        ),
                        "auroc": float(metadata["auroc"]),
                        "tpr_at_fpr": float(metadata["tpr_at_fpr"]),
                        "path": path,
                    }
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                invalid_files.append(f"{path}: {exc}")

        if invalid_files:
            details = "\n".join(invalid_files)
            raise ValueError(f"Invalid NPZ result metadata:\n{details}")
        if not rows:
            raise FileNotFoundError(f"No .npz result files found under {self.results_dir}.")

        return pd.DataFrame(rows)

    def plot_heatmap(
        self,
        detector_name=None,
        target_regime=None,
        metrics=("auroc", "tpr_at_fpr"),
        figsize=None,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        output_path=None,
        formats=None,
    ):
        """
        Plot generator-by-dataset heatmaps from all NPZ results.

        ``detector_name`` and ``target_regime`` select a single experimental
        slice. If multiple results still map to the same cell, their metric is
        averaged. The returned DataFrame contains the filtered, unaggregated
        rows so this behavior remains easy to inspect.
        """
        frame = self.load_npz_metrics()
        frame = self._filter_frame(
            frame,
            {"detector": detector_name, "target_regime": target_regime},
        )
        title_parts = [value for value in (detector_name, target_regime) if value]
        title = " · ".join(map(str, title_parts)) or None
        return self._plot_metric_heatmaps(
            frame,
            index="generator",
            ylabel="Generator",
            metrics=metrics,
            figsize=figsize,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            title=title,
            output_path=output_path,
            formats=formats,
        )

    def plot_global_heatmap(
        self,
        target_regime=None,
        metrics=("auroc", "tpr_at_fpr"),
        figsize=None,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        output_path=None,
        formats=None,
    ):
        """
        Plot detector-by-dataset heatmaps aggregated across generators.

        By default all target regimes are included. Pass ``target_regime`` to
        restrict the view; repeated detector-by-dataset cells are averaged.
        """
        frame = self.load_npz_metrics()
        frame = self._filter_frame(frame, {"target_regime": target_regime})
        title = f"Global · {target_regime}" if target_regime else "Global"
        return self._plot_metric_heatmaps(
            frame,
            index="detector",
            ylabel="Detector",
            metrics=metrics,
            figsize=figsize,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            title=title,
            output_path=output_path,
            formats=formats,
        )

    def plot_grid_heatmap(
        self,
        target_regime=None,
        detector_names=None,
        metrics=("auroc", "tpr_at_fpr"),
        ncols=2,
        figsize=None,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        output_path=None,
        formats=None,
    ):
        """
        Plot a grid of detector heatmaps in one separate figure per metric.

        Each panel is a generator-by-dataset heatmap for one detector. With the
        default metrics this creates two figures, one for AUROC and one for
        TPR@1% FPR. All panels use the same 0--1 colour scale and a shared
        colourbar, making detector values directly comparable.

        ``detector_names`` can select detectors and control their panel order.
        The default two-column layout uses an A4 portrait figure; layouts with
        three or more columns use A4 landscape. ``output_path`` is treated as a
        filename stem and the metric name is appended to each saved figure.

        Returns ``(figures, axes, frame)``, where the first two values are
        dictionaries keyed by metric and ``frame`` contains the filtered,
        unaggregated result rows.
        """
        plt, sns = self._require_plotting()
        self.ensure_style()

        if not isinstance(ncols, int) or isinstance(ncols, bool) or ncols < 1:
            raise ValueError("ncols must be a positive integer.")

        metrics = tuple(metrics)
        labels = {"auroc": "AUROC", "tpr_at_fpr": "TPR@1% FPR"}
        unknown = set(metrics) - set(labels)
        if unknown:
            raise ValueError(f"Unsupported heatmap metrics: {sorted(unknown)}")
        if not metrics:
            raise ValueError("At least one heatmap metric is required.")

        frame = self.load_npz_metrics()
        frame = self._filter_frame(frame, {"target_regime": target_regime})

        if detector_names is None:
            detectors = sorted(frame["detector"].dropna().unique(), key=str)
        else:
            if isinstance(detector_names, str):
                detector_names = [detector_names]
            detectors = list(dict.fromkeys(detector_names))
            if not detectors:
                raise ValueError("detector_names must contain at least one detector.")
            available = set(frame["detector"].dropna())
            missing = [name for name in detectors if name not in available]
            if missing:
                raise ValueError(
                    "No results match the selected target regime for detectors: "
                    f"{missing}"
                )
            frame = frame[frame["detector"].isin(detectors)]

        if frame.empty or not detectors:
            raise ValueError("No results match the selected filters.")

        generators = sorted(frame["generator"].dropna().unique(), key=str)
        datasets = sorted(frame["dataset"].dropna().unique(), key=str)
        nrows = int(np.ceil(len(detectors) / ncols))
        if figsize is None:
            # ISO A4 dimensions in inches, switching orientation for 3xY grids.
            figsize = (8.27, 11.69) if ncols <= 2 else (11.69, 8.27)

        figures = {}
        axes_by_metric = {}
        for metric in metrics:
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=figsize,
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
                    sort=True,
                ).reindex(index=generators, columns=datasets)

                sns.heatmap(
                    matrix,
                    ax=ax,
                    cmap=cmap,
                    vmin=0.0,
                    vmax=1.0,
                    annot=annot,
                    fmt=fmt,
                    linewidths=0.4,
                    linecolor="0.85",
                    cbar=False,
                    annot_kws={"fontsize": 6},
                )
                self.format_axis(
                    ax,
                    xlabel="Dataset"
                    if panel_idx + ncols >= len(detectors)
                    else "",
                    ylabel="Generator" if panel_idx % ncols == 0 else "",
                    title=str(detector),
                    grid_axis=None,
                )
                ax.tick_params(axis="x", rotation=35)
                ax.tick_params(axis="y", rotation=0)
                visible_axes.append(ax)

            for ax in flat_axes[len(detectors) :]:
                ax.set_visible(False)

            colorbar = fig.colorbar(
                visible_axes[0].collections[0],
                ax=visible_axes,
                location="right",
                shrink=0.8,
                pad=0.02,
            )
            colorbar.set_label(labels[metric])
            title = labels[metric]
            if target_regime is not None:
                title = f"{title} · {target_regime}"
            fig.suptitle(title)

            if output_path is not None:
                metric_path = Path(output_path).with_name(
                    f"{Path(output_path).stem}_{metric}"
                )
                self.save_figure(fig, metric_path, formats=formats)

            figures[metric] = fig
            axes_by_metric[metric] = axes

        return figures, axes_by_metric, frame

    def plot_detection_gap_barplot(
        self,
        detector_names=None,
        regimes=("free_llm", "llm2l", "h2l"),
        n_bootstrap=5000,
        random_seed=42,
        alpha=0.05,
        figsize=(7.1, 3.6),
        title="Figure 3 — H2L vs LLM2L detection gap",
        output_path=None,
        formats=None,
        hatches=("", "///", "xx"),
    ):
        """Plot detector-level mean TPR@1% FPR with bootstrap confidence intervals.

        Predictions are bootstrapped within each dataset-by-generator block,
        then the block results are averaged at detector level. One grouped bar
        is drawn per regime and asymmetric percentile confidence intervals are
        shown using ``alpha`` (95% by default).

        Returns ``(fig, ax, summary_frame)``. The summary contains one row per
        detector and regime with the plotted mean, CI bounds, and block count.
        """
        from utils.evaluation.BootstrapMetrics import bootstrap_metrics_for_detector
        from utils.evaluation.Metrics import (
            Metrics,
            filter_metrics_by_target_regime,
        )

        plt, sns = self._require_plotting()
        self.ensure_style()

        if not isinstance(n_bootstrap, int) or isinstance(n_bootstrap, bool) or n_bootstrap < 1:
            raise ValueError("n_bootstrap must be a positive integer.")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1.")

        regimes = tuple(regimes)
        regime_labels = {
            "free_llm": "FreeLLM",
            "llm2l": "LLM2L",
            "h2l": "H2L",
        }
        unknown_regimes = set(regimes) - set(regime_labels)
        if unknown_regimes:
            raise ValueError(f"Unsupported detection regimes: {sorted(unknown_regimes)}")
        if not regimes:
            raise ValueError("At least one detection regime is required.")

        available_frame = self.load_npz_metrics()
        if detector_names is None:
            detectors = sorted(available_frame["detector"].dropna().unique(), key=str)
        else:
            if isinstance(detector_names, str):
                detector_names = [detector_names]
            detectors = list(dict.fromkeys(detector_names))
        if not detectors:
            raise ValueError("detector_names must contain at least one detector.")

        available_detectors = set(available_frame["detector"].dropna())
        missing_detectors = [name for name in detectors if name not in available_detectors]
        if missing_detectors:
            raise ValueError(f"No results found for detectors: {missing_detectors}")

        rows = []
        for detector in detectors:
            detector_metrics = Metrics.load_metrics_of_detector(self.results_dir, detector)
            for regime in regimes:
                regime_metrics = filter_metrics_by_target_regime(detector_metrics, regime)
                if not regime_metrics:
                    raise ValueError(f"No {regime} results found for detector {detector}.")

                bootstrap = bootstrap_metrics_for_detector(
                    regime_metrics,
                    n_bootstrap=n_bootstrap,
                    random_seed=random_seed,
                )
                summary = bootstrap.summary(alpha=alpha)
                rows.append(
                    {
                        "detector": detector,
                        "target_regime": regime,
                        "regime": regime_labels[regime],
                        "mean_tpr_at_fpr": summary["tpr_at_fpr_mean"],
                        "tpr_ci_low": summary["tpr_at_fpr_ci_low"],
                        "tpr_ci_high": summary["tpr_at_fpr_ci_high"],
                        "n_blocks": len(regime_metrics),
                        "n_bootstrap": n_bootstrap,
                        "alpha": alpha,
                    }
                )

        summary_frame = pd.DataFrame(rows)
        hue_order = [regime_labels[regime] for regime in regimes]
        palette = self.get_colorblind_palette(len(hue_order))
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(
            data=summary_frame,
            x="detector",
            y="mean_tpr_at_fpr",
            hue="regime",
            order=detectors,
            hue_order=hue_order,
            palette=palette,
            estimator="mean",
            errorbar=None,
            edgecolor="0.15",
            linewidth=0.5,
            ax=ax,
        )
        self.apply_bar_hatches(ax, hatches=hatches, hue_levels=hue_order)
        # self._add_ci_error_bars(
        #     ax,
        #     summary_frame,
        #     x="detector",
        #     y="mean_tpr_at_fpr",
        #     hue="regime",
        #     order=detectors,
        #     hue_order=hue_order,
        #     ci_low="tpr_ci_low",
        #     ci_high="tpr_ci_high",
        # )
        self.format_axis(
            ax,
            xlabel="Detector",
            ylabel="Mean TPR@1% FPR",
            title=title,
            grid_axis="y",
        )
        ax.set_ylim(0.0, 1.0)
        ax.tick_params(axis="x", rotation=20)
        ax.legend(
            title=None,
            frameon=False,
            ncols=len(hue_order),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
        )
        ax.set_title(title, pad=31)
        fig.tight_layout(pad=0.3)

        if output_path is not None:
            self.save_figure(fig, output_path, formats=formats)
        return fig, ax, summary_frame

    def _plot_metric_heatmaps(
        self,
        frame,
        index,
        ylabel,
        metrics,
        figsize,
        annot,
        fmt,
        cmap,
        title,
        output_path,
        formats,
    ):
        """Render one dataset heatmap per metric from a filtered result frame."""
        plt, sns = self._require_plotting()
        self.ensure_style()

        if frame.empty:
            raise ValueError("No results match the selected filters.")

        metrics = tuple(metrics)
        labels = {"auroc": "AUROC", "tpr_at_fpr": "TPR@1% FPR"}
        unknown = set(metrics) - set(labels)
        if unknown:
            raise ValueError(f"Unsupported heatmap metrics: {sorted(unknown)}")
        if not metrics:
            raise ValueError("At least one heatmap metric is required.")

        n_datasets = frame["dataset"].nunique()
        n_rows = frame[index].nunique()
        if figsize is None:
            figsize = (
                max(3.2 * len(metrics), 1.25 * n_datasets * len(metrics)),
                max(2.4, 0.55 * n_rows + 1.2),
            )

        fig, axes = plt.subplots(
            1,
            len(metrics),
            figsize=figsize,
            squeeze=False,
            sharey=True,
        )
        axes = axes.ravel()

        for ax, metric in zip(axes, metrics):
            matrix = frame.pivot_table(
                index=index,
                columns="dataset",
                values=metric,
                aggfunc="mean",
                sort=True,
            )
            sns.heatmap(
                matrix,
                ax=ax,
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                annot=annot,
                fmt=fmt,
                linewidths=0.4,
                linecolor="0.85",
                cbar_kws={"label": labels[metric]},
                annot_kws={"fontsize": 7},
            )
            self.format_axis(
                ax,
                xlabel="Dataset",
                ylabel=ylabel if ax is axes[0] else None,
                title=labels[metric],
                grid_axis=None,
            )
            ax.tick_params(axis="x", rotation=35)
            ax.tick_params(axis="y", rotation=0)

        if title:
            fig.suptitle(title, y=1.02)
        fig.tight_layout(pad=0.4)

        if output_path is not None:
            self.save_figure(fig, output_path, formats=formats)
        return fig, axes, frame

    def plot_grouped_barplot(
        self,
        data,
        x,
        y,
        hue,
        output_path,
        title,
        ylabel,
        xlabel=None,
        figsize=(7.1, 3.2),
        order=None,
        hue_order=None,
        ci_low=None,
        ci_high=None,
        hatches=None,
        formats=None,
        baseline=None,
        ylim=None,
        x_rotation=25,
        legend_title=None,
        legend_ncols=None,
        value_labels=False,
    ):
        """
        Plot grouped bars and save them.

        Hatch patterns supplement color so the groups remain distinguishable in
        grayscale printouts. ``baseline`` is optional so the same renderer can
        be used for both signed deltas and bounded metrics.
        """
        plt, sns = self._require_plotting()
        self.ensure_style()
        frame = self._as_frame(data)
        missing = {x, y, hue} - set(frame.columns)
        if missing:
            raise ValueError(f"Missing barplot columns: {sorted(missing)}")

        order = list(order or self._ordered_unique(frame[x]))
        hue_order = list(hue_order or self._ordered_unique(frame[hue]))
        palette = self.get_colorblind_palette(len(hue_order))

        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(
            data=frame,
            x=x,
            y=y,
            hue=hue,
            order=order,
            hue_order=hue_order,
            palette=palette,
            estimator="mean",
            errorbar=None,
            edgecolor="0.15",
            linewidth=0.5,
            ax=ax,
        )
        self.apply_bar_hatches(ax, hatches=hatches, hue_levels=hue_order)
        if baseline is not None:
            ax.axhline(float(baseline), color="0.15", linewidth=0.8)
        # self._add_ci_error_bars(ax, frame, x, y, hue, order, hue_order, ci_low, ci_high)
        self.format_axis(ax, xlabel=xlabel or x.title(), ylabel=ylabel, title=title, grid_axis="y")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.tick_params(axis="x", rotation=x_rotation)
        ax.legend(
            title=legend_title,
            frameon=False,
            ncols=legend_ncols or min(3, max(1, len(hue_order))),
        )
        if value_labels:
            for container in [item for item in ax.containers if hasattr(item, "patches")]:
                ax.bar_label(container, fmt="%.3f", padding=2, fontsize=7)
        fig.tight_layout(pad=0.2)
        self.save_figure(fig, output_path, formats=formats)
        return fig, ax

    def plot_grouped_delta_barplot(
        self,
        data,
        x,
        y,
        hue,
        output_path,
        title,
        ylabel,
        xlabel=None,
        figsize=(7.1, 3.2),
        order=None,
        hue_order=None,
        ci_low=None,
        ci_high=None,
        hatches=None,
        formats=None,
    ):
        """Plot signed grouped deltas using the shared grouped-bar renderer."""
        return self.plot_grouped_barplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            output_path=output_path,
            title=title,
            ylabel=ylabel,
            xlabel=xlabel,
            figsize=figsize,
            order=order,
            hue_order=hue_order,
            ci_low=ci_low,
            ci_high=ci_high,
            hatches=hatches,
            formats=formats,
            baseline=0.0,
            x_rotation=25,
            legend_title=hue.title(),
        )

    def _add_ci_error_bars(self, ax, frame, x, y, hue, order, hue_order, ci_low, ci_high) -> None:
        ci_low, ci_high = self._resolve_ci_columns(frame, y, ci_low, ci_high)
        if ci_low is None or ci_high is None:
            return

        containers = [container for container in ax.containers if hasattr(container, "patches")]
        for hue_idx, container in enumerate(containers[: len(hue_order)]):
            hue_value = hue_order[hue_idx]
            for x_idx, patch in enumerate(container.patches[: len(order)]):
                x_value = order[x_idx]
                rows = frame[(frame[x] == x_value) & (frame[hue] == hue_value)]
                if rows.empty:
                    continue
                center = float(rows[y].mean())
                low = float(rows[ci_low].mean())
                high = float(rows[ci_high].mean())
                x_center = patch.get_x() + patch.get_width() / 2
                ax.errorbar(
                    x_center,
                    center,
                    yerr=[[max(center - low, 0.0)], [max(high - center, 0.0)]],
                    color="0.15",
                    capsize=2,
                    linewidth=0.7,
                    fmt="none",
                    zorder=5,
                )

    def plot_metric_lines(
        self,
        data,
        x,
        y,
        hue,
        output_path,
        title,
        ylabel,
        xlabel=None,
        figsize=(7.1, 3.2),
        hue_order=None,
        formats=None,
    ):
        """Plot metric curves using color plus linestyle and markers."""
        plt, _ = self._require_plotting()
        self.ensure_style()
        frame = self._as_frame(data)
        missing = {x, y, hue} - set(frame.columns)
        if missing:
            raise ValueError(f"Missing lineplot columns: {sorted(missing)}")

        hue_order = list(hue_order or self._ordered_unique(frame[hue]))
        palette = self.get_colorblind_palette(len(hue_order))
        fig, ax = plt.subplots(figsize=figsize)

        for idx, hue_value in enumerate(hue_order):
            subset = frame[frame[hue] == hue_value].sort_values(x)
            ax.plot(
                subset[x],
                subset[y],
                label=str(hue_value),
                color=palette[idx],
                linestyle=DEFAULT_LINESTYLES[idx % len(DEFAULT_LINESTYLES)],
                marker=DEFAULT_MARKERS[idx % len(DEFAULT_MARKERS)],
            )

        self.format_axis(ax, xlabel=xlabel or x.title(), ylabel=ylabel, title=title, grid_axis="y")
        ax.legend(title=hue.title(), frameon=False)
        fig.tight_layout(pad=0.2)
        self.save_figure(fig, output_path, formats=formats)
        return fig, ax

    def load_metric_reports(self, filename: str) -> pd.DataFrame:
        """Load matching per-run metric CSV files from the results directory."""
        frames = []
        for path in sorted(self.results_dir.glob(f"*/{filename}")):
            frame = pd.read_csv(path)
            frame["result_dir"] = path.parent.name
            frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def load_ranking_metrics(self) -> pd.DataFrame:
        frame = self.load_metric_reports("ranking_metrics.csv")
        legacy_column = "tpr_at_1pct_fpr"
        if legacy_column in frame.columns:
            if "tpr_at_fpr" not in frame.columns:
                frame["tpr_at_fpr"] = frame[legacy_column]
            else:
                frame["tpr_at_fpr"] = frame["tpr_at_fpr"].fillna(frame[legacy_column])
            frame = frame.drop(columns=[legacy_column])
        return frame

    def load_threshold_transfer_metrics(self) -> pd.DataFrame:
        return self.load_metric_reports("threshold_transfer_metrics.csv")

    def plot_ranking_metric_heatmap(
        self,
        metric="tpr_at_fpr",
        target_regime="h2l",
        output_path=None,
        title=None,
        cbar_label=None,
        filters: Mapping[str, object] | None = None,
        formats=None,
    ):
        frame = self.load_ranking_metrics()
        if frame.empty:
            raise FileNotFoundError(f"No ranking_metrics.csv files found under {self.results_dir}.")
        frame = self._filter_frame(frame, {"target_regime": target_regime, **(filters or {})})
        output_path = output_path or self.output_dir / f"ranking_{metric}_{target_regime}_heatmap"
        title = title or f"{metric} for {target_regime}"
        cbar_label = cbar_label or metric
        return self.plot_delta_heatmap(
            frame,
            output_path=output_path,
            title=title,
            cbar_label=cbar_label,
            values=metric,
            center=0.0 if metric.startswith("delta_") else None,
            formats=formats,
        )

    def plot_threshold_transfer_delta_heatmap(
        self,
        metric="delta_tpr_vs_free_llm",
        target_regime="h2l",
        output_path=None,
        title=None,
        cbar_label=None,
        filters: Mapping[str, object] | None = None,
        formats=None,
    ):
        frame = self.load_threshold_transfer_metrics()
        if frame.empty:
            raise FileNotFoundError(
                f"No threshold_transfer_metrics.csv files found under {self.results_dir}."
            )
        frame = self._filter_frame(frame, {"target_regime": target_regime, **(filters or {})})
        output_path = output_path or self.output_dir / f"transfer_{metric}_{target_regime}_heatmap"
        title = title or f"{metric} for {target_regime}"
        cbar_label = cbar_label or metric
        return self.plot_delta_heatmap(
            frame,
            output_path=output_path,
            title=title,
            cbar_label=cbar_label,
            values=metric,
            center=0.0,
            formats=formats,
        )

    def plot_threshold_transfer_delta_barplot(
        self,
        metric="delta_tpr_vs_free_llm",
        output_path=None,
        title=None,
        filters: Mapping[str, object] | None = None,
        formats=None,
    ):
        frame = self.load_threshold_transfer_metrics()
        if frame.empty:
            raise FileNotFoundError(
                f"No threshold_transfer_metrics.csv files found under {self.results_dir}."
            )
        frame = self._filter_frame(frame, filters or {})
        output_path = output_path or self.output_dir / f"transfer_{metric}_barplot"
        title = title or f"{metric} by detector and regime"
        return self.plot_grouped_delta_barplot(
            frame,
            x="detector",
            y=metric,
            hue="target_regime",
            output_path=output_path,
            title=title,
            ylabel=metric,
            formats=formats,
        )

    @staticmethod
    def _filter_frame(frame: pd.DataFrame, filters: Mapping[str, object]) -> pd.DataFrame:
        result = frame.copy()
        for column, value in filters.items():
            if value is None:
                continue
            if column not in result.columns:
                raise ValueError(f"Cannot filter on missing column: {column}")
            if isinstance(value, (list, tuple, set, frozenset)):
                result = result[result[column].isin(value)]
            else:
                result = result[result[column] == value]
        return result


def plot_heatmap(results_dir, **kwargs):
    """Convenience wrapper for :meth:`ResultsReportGraphBuilder.plot_heatmap`."""
    return ResultsReportGraphBuilder(results_dir=results_dir).plot_heatmap(**kwargs)


def plot_grid_heatmap(results_dir, **kwargs):
    """Plot one grid of detector heatmaps per metric from NPZ results."""
    return ResultsReportGraphBuilder(results_dir=results_dir).plot_grid_heatmap(**kwargs)


def plot_detection_gap_barplot(results_dir, **kwargs):
    """Plot the detector-level FreeLLM, LLM2L, and H2L TPR comparison."""
    return ResultsReportGraphBuilder(results_dir=results_dir).plot_detection_gap_barplot(**kwargs)


def plot_global_heatmap(results_dir, **kwargs):
    """Plot global detector-by-dataset heatmaps from NPZ results."""
    return ResultsReportGraphBuilder(results_dir=results_dir).plot_global_heatmap(**kwargs)


__all__ = [
    "ResultsReportGraphBuilder",
    "plot_detection_gap_barplot",
    "plot_global_heatmap",
    "plot_grid_heatmap",
    "plot_heatmap",
]
