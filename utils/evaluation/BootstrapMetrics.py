from typing import List

import numpy as np

from utils.evaluation.Metrics import (
    Metrics,
    compute_auroc,
    compute_tpr_at_fpr,
    get_prediction_scores,
)


class BootstrapMetrics:
    """
    Bootstrap summaries for the current evaluation design.
    """

    def __init__(
        self,
        auroc_array,
        tpr_at_fpr_array,
        tpr_fpr_target=0.01,
    ):
        self.auroc_array = np.array(auroc_array, dtype=float)
        self.tpr_at_fpr_array = np.array(tpr_at_fpr_array, dtype=float)
        self.tpr_fpr_target = tpr_fpr_target

    @staticmethod
    def _nanmean(values):
        return float(np.nanmean(values))

    @staticmethod
    def _nanstd(values):
        return float(np.nanstd(values))

    @property
    def auroc_mean(self):
        return self._nanmean(self.auroc_array)

    @property
    def auroc_std(self):
        return self._nanstd(self.auroc_array)

    @property
    def tpr_at_fpr_mean(self):
        return self._nanmean(self.tpr_at_fpr_array)

    @property
    def tpr_at_fpr_std(self):
        return self._nanstd(self.tpr_at_fpr_array)

    def summary(self, alpha=0.05):
        auroc_low, auroc_high = self.ci_bounds(self.auroc_array, alpha=alpha)
        tpr_low, tpr_high = self.ci_bounds(self.tpr_at_fpr_array, alpha=alpha)

        return {
            "auroc_mean": self.auroc_mean,
            "auroc_std": self.auroc_std,
            "auroc_ci_low": auroc_low,
            "auroc_ci_high": auroc_high,
            "auroc_ci_halfwidth": (auroc_high - auroc_low) / 2,

            "tpr_at_fpr_mean": self.tpr_at_fpr_mean,
            "tpr_at_fpr_std": self.tpr_at_fpr_std,
            "tpr_at_fpr_ci_low": tpr_low,
            "tpr_at_fpr_ci_high": tpr_high,
            "tpr_at_fpr_ci_halfwidth": (tpr_high - tpr_low) / 2,

            "tpr_fpr_target": self.tpr_fpr_target,
        }

    def ci_bounds(self, metric_array=None, alpha=0.05):
        if metric_array is None:
            metric_array = self.auroc_array

        lower = np.nanpercentile(metric_array, 100 * alpha / 2)
        upper = np.nanpercentile(metric_array, 100 * (1 - alpha / 2))

        return float(lower), float(upper)

    def ci_halfwidth(self, metric_array=None, alpha=0.05):
        if metric_array is None:
            metric_array = self.auroc_array
        lower = np.nanpercentile(metric_array, 100 * alpha / 2)
        upper = np.nanpercentile(metric_array, 100 * (1 - alpha / 2))
        return float((upper - lower) / 2)

    def ci_width(self, metric_array=None, alpha=0.05):
        if metric_array is None:
            metric_array = self.auroc_array
        lower = np.nanpercentile(metric_array, 100 * alpha / 2)
        upper = np.nanpercentile(metric_array, 100 * (1 - alpha / 2))
        return float(upper - lower)

    def required_sample_size(self, w_target, metric_array=None, alpha=0.05):
        if metric_array is None:
            metric_array = self.auroc_array
        n_pilot = len(metric_array)
        w_pilot = self.ci_halfwidth(metric_array, alpha)
        n_target = n_pilot * (w_pilot / w_target) ** 2
        return int(np.ceil(n_target))


def bootstrap_metrics(metrics: Metrics, n_bootstrap=1000, random_seed=42):
    """
    Performs bootstrap sampling on predictions and computes AUROC and TPR@FPR.
    """
    rng = np.random.default_rng(random_seed)
    n = len(metrics.preds.true_labels)
    aurocs, tprs = [], []

    true_labels = np.array(metrics.preds.true_labels)
    scores = np.array(get_prediction_scores(metrics.preds))

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        sampled_true = true_labels[idx]
        sampled_scores = scores[idx]

        aurocs.append(compute_auroc(sampled_true, sampled_scores))
        tprs.append(
            compute_tpr_at_fpr(
                sampled_true,
                sampled_scores,
                target_fpr=metrics.tpr_fpr_target,
            )
        )

    return BootstrapMetrics(
        auroc_array=aurocs,
        tpr_at_fpr_array=tprs,
        tpr_fpr_target=metrics.tpr_fpr_target,
    )



def bootstrap_metrics_for_detector(
    metrics: List[Metrics],
    n_bootstrap=5000,
    random_seed=42,
):
    """
    Input:
        lista di Metrics dello stesso detector,
        su più dataset × generator × comparison.

    Output:
        BootstrapMetrics aggregato a livello detector.
    """
    auroc_arrays = []
    tpr_at_fpr_arrays = []

    for i, m in enumerate(metrics):
        bm = bootstrap_metrics(
            m,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed + i,
        )
        auroc_arrays.append(bm.auroc_array)
        tpr_at_fpr_arrays.append(bm.tpr_at_fpr_array)

    auroc_mat = np.vstack(auroc_arrays)
    tpr_mat = np.vstack(tpr_at_fpr_arrays)

    aggregated_auroc = np.nanmean(auroc_mat, axis=0)
    aggregated_tpr = np.nanmean(tpr_mat, axis=0)

    return BootstrapMetrics(
        auroc_array=aggregated_auroc,
        tpr_at_fpr_array=aggregated_tpr,
        tpr_fpr_target=metrics[0].tpr_fpr_target if metrics else 0.01,
    )

def normalize_pairing_ids(ids):
    """
    Normalizza gli ids per confronti paired tra baseline e target.

    Esempi:
        human::12    -> human::12
        free_llm::12 -> machine::12
        freellm::12  -> machine::12
        h2l::12      -> machine::12
        llm2l::12    -> machine::12
    """
    normalized = []

    for x in ids:
        x = str(x)

        if "::" not in x:
            normalized.append(x)
            continue

        prefix, idx = x.split("::", 1)

        if prefix in {"free_llm", "freellm", "llmfree", "h2l", "llm2l"}:
            normalized.append(f"machine::{idx}")
        elif prefix == "human":
            normalized.append(f"human::{idx}")
        else:
            normalized.append(x)

    return np.asarray(normalized)


def check_pairing_alignment(baseline_metrics, target_metrics, n_preview=5):
    baseline_ids = np.asarray(getattr(baseline_metrics.preds, "ids", []))
    target_ids = np.asarray(getattr(target_metrics.preds, "ids", []))

    baseline_pair_ids = normalize_pairing_ids(baseline_ids)
    target_pair_ids = normalize_pairing_ids(target_ids)

    print("Baseline first:", baseline_ids[:n_preview])
    print("Target first:  ", target_ids[:n_preview])
    print("Norm base first:", baseline_pair_ids[:n_preview])
    print("Norm targ first:", target_pair_ids[:n_preview])

    print("Baseline last:", baseline_ids[-n_preview:])
    print("Target last:  ", target_ids[-n_preview:])
    print("Norm base last:", baseline_pair_ids[-n_preview:])
    print("Norm targ last:", target_pair_ids[-n_preview:])

    print("Raw ids equal:", np.array_equal(baseline_ids, target_ids))
    print("Normalized ids equal:", np.array_equal(baseline_pair_ids, target_pair_ids))


def bootstrap_paired_delta_for_block(
    baseline_metrics,
    target_metrics,
    metric_name,
    n_bootstrap=5000,
    random_seed=42,
) -> np.ndarray:
    rng = np.random.default_rng(random_seed)

    baseline_true = np.asarray(baseline_metrics.preds.true_labels)
    target_true = np.asarray(target_metrics.preds.true_labels)

    if len(baseline_true) != len(target_true):
        raise ValueError("Baseline and target have different lengths.")

    if not np.array_equal(baseline_true, target_true):
        raise ValueError("Baseline and target true_labels are not aligned.")

    baseline_scores = np.asarray(get_prediction_scores(baseline_metrics.preds), dtype=float)
    target_scores = np.asarray(get_prediction_scores(target_metrics.preds), dtype=float)

    if len(baseline_scores) != len(target_scores):
        raise ValueError("Baseline and target scores have different lengths.")

    baseline_ids = np.asarray(getattr(baseline_metrics.preds, "ids", []))
    target_ids = np.asarray(getattr(target_metrics.preds, "ids", []))

    if baseline_ids.size > 0 and target_ids.size > 0:
        if len(baseline_ids) != len(target_ids):
            raise ValueError("Baseline and target ids have different lengths.")

        baseline_pair_ids = normalize_pairing_ids(baseline_ids)
        target_pair_ids = normalize_pairing_ids(target_ids)

        if not np.array_equal(baseline_pair_ids, target_pair_ids):
            raise ValueError(
                "Baseline and target ids are not aligned after pairing normalization."
            )

    n = len(baseline_true)
    deltas = np.empty(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, n)

        y = baseline_true[idx]
        s_base = baseline_scores[idx]
        s_target = target_scores[idx]

        if metric_name == "auroc":
            base_value = compute_auroc(y, s_base)
            target_value = compute_auroc(y, s_target)

        elif metric_name == "tpr_at_fpr":
            base_value = compute_tpr_at_fpr(
                y,
                s_base,
                target_fpr=baseline_metrics.tpr_fpr_target,
            )
            target_value = compute_tpr_at_fpr(
                y,
                s_target,
                target_fpr=baseline_metrics.tpr_fpr_target,
            )

        else:
            raise ValueError(f"Unsupported metric '{metric_name}'.")

        deltas[b] = target_value - base_value

    return deltas

def bootstrap_paired_delta_for_detector(
    baseline_metrics: List[Metrics],
    target_metrics: List[Metrics],
    metric_name: str,
    n_bootstrap=5000,
    random_seed=42,
) -> np.ndarray:
    """
    Aggrega delta bootstrap a livello detector.

    Input:
        baseline_metrics: lista di Metrics baseline HUMAN vs FREE_LLM
        target_metrics: lista di Metrics target HUMAN vs H2L o HUMAN vs LLM2L

    Ogni elemento delle due liste deve corrispondere allo stesso:
        detector × dataset × generator

    Output:
        array bootstrap aggregato, shape = (n_bootstrap,)
    """
    if len(baseline_metrics) != len(target_metrics):
        raise ValueError(
            f"Different number of blocks: "
            f"baseline={len(baseline_metrics)}, target={len(target_metrics)}"
        )

    delta_arrays = []

    for i, (base_m, target_m) in enumerate(zip(baseline_metrics, target_metrics)):
        delta = bootstrap_paired_delta_for_block(
            baseline_metrics=base_m,
            target_metrics=target_m,
            metric_name=metric_name,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed + i,
        )
        delta_arrays.append(delta)

    delta_mat = np.vstack(delta_arrays)

    # Per ogni bootstrap iteration, media sui blocchi dataset × generator.
    aggregated_delta = np.nanmean(delta_mat, axis=0)

    return aggregated_delta


def summarize_bootstrap_array(array, alpha=0.05):
    array = np.asarray(array, dtype=float)

    low = np.nanpercentile(array, 100 * alpha / 2)
    high = np.nanpercentile(array, 100 * (1 - alpha / 2))

    return {
        "mean": float(np.nanmean(array)),
        "std": float(np.nanstd(array)),
        "ci_low": float(low),
        "ci_high": float(high),
        "ci_halfwidth": float((high - low) / 2),
    }