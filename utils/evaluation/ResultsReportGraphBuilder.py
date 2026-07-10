from __future__ import annotations

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
        """
        Plot grouped delta bars and save them.

        Hatch patterns supplement color so the groups remain distinguishable in
        grayscale printouts.
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
        ax.axhline(0.0, color="0.15", linewidth=0.8)
        self._add_ci_error_bars(ax, frame, x, y, hue, order, hue_order, ci_low, ci_high)
        self.format_axis(ax, xlabel=xlabel or x.title(), ylabel=ylabel, title=title, grid_axis="y")
        ax.tick_params(axis="x", rotation=25)
        ax.legend(title=hue.title(), frameon=False, ncols=min(3, max(1, len(hue_order))))
        fig.tight_layout(pad=0.2)
        self.save_figure(fig, output_path, formats=formats)
        return fig, ax

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


__all__ = ["ResultsReportGraphBuilder"]
