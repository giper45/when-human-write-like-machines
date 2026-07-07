import numpy as np

from utils.evaluation.Metrics import (
    Metrics,
    compute_auroc,
    compute_brier_score,
    compute_ece,
    compute_tpr_at_fpr,
)


class BootstrapMetrics:
    """
    Bootstrap summaries for the current evaluation design.
    """

    def __init__(
        self,
        auroc_array,
        tpr_at_fpr_array,
        ece_array,
        brier_array,
        tpr_fpr_target=0.01,
    ):
        self.auroc_array = np.array(auroc_array, dtype=float)
        self.tpr_at_fpr_array = np.array(tpr_at_fpr_array, dtype=float)
        self.ece_array = np.array(ece_array, dtype=float)
        self.brier_array = np.array(brier_array, dtype=float)
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

    @property
    def ece_mean(self):
        return self._nanmean(self.ece_array)

    @property
    def ece_std(self):
        return self._nanstd(self.ece_array)

    @property
    def brier_mean(self):
        return self._nanmean(self.brier_array)

    @property
    def brier_std(self):
        return self._nanstd(self.brier_array)

    def summary(self):
        return {
            "auroc_mean": self.auroc_mean,
            "auroc_std": self.auroc_std,
            "tpr_at_fpr_mean": self.tpr_at_fpr_mean,
            "tpr_at_fpr_std": self.tpr_at_fpr_std,
            "tpr_fpr_target": self.tpr_fpr_target,
            "ece_mean": self.ece_mean,
            "ece_std": self.ece_std,
            "brier_mean": self.brier_mean,
            "brier_std": self.brier_std,
        }

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
    Performs bootstrap sampling on predictions and computes AUROC, TPR@FPR, ECE, and Brier.
    """
    rng = np.random.default_rng(random_seed)
    n = len(metrics.preds.true_labels)
    aurocs, tprs, eces, briers = [], [], [], []

    true_labels = np.array(metrics.preds.true_labels)
    pred_probs = np.array(metrics.preds.pred_probs)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        sampled_true = true_labels[idx]
        sampled_probs = pred_probs[idx]

        aurocs.append(compute_auroc(sampled_true, sampled_probs))
        tprs.append(
            compute_tpr_at_fpr(
                sampled_true,
                sampled_probs,
                target_fpr=metrics.tpr_fpr_target,
            )
        )
        eces.append(compute_ece(sampled_true, sampled_probs, n_bins=metrics.ece_bins))
        briers.append(compute_brier_score(sampled_true, sampled_probs))

    return BootstrapMetrics(
        auroc_array=aurocs,
        tpr_at_fpr_array=tprs,
        ece_array=eces,
        brier_array=briers,
        tpr_fpr_target=metrics.tpr_fpr_target,
    )


def bootstrap_metric_delta(metrics1, metrics2, metric_name="auroc", n_bootstrap=1000, random_seed=42):
    """
    Returns bootstrap deltas for a selected metric.
    """
    bm1 = bootstrap_metrics(metrics1, n_bootstrap, random_seed)
    bm2 = bootstrap_metrics(metrics2, n_bootstrap, random_seed)

    metric_map_1 = {
        "auroc": bm1.auroc_array,
        "tpr_at_fpr": bm1.tpr_at_fpr_array,
        "ece": bm1.ece_array,
        "brier": bm1.brier_array,
    }
    metric_map_2 = {
        "auroc": bm2.auroc_array,
        "tpr_at_fpr": bm2.tpr_at_fpr_array,
        "ece": bm2.ece_array,
        "brier": bm2.brier_array,
    }

    if metric_name not in metric_map_1:
        raise ValueError(f"Unsupported metric '{metric_name}'.")

    return metric_map_1[metric_name] - metric_map_2[metric_name]


def bootstrap_delta_auroc(metrics1, metrics2, n_bootstrap=1000, random_seed=42):
    return bootstrap_metric_delta(
        metrics1,
        metrics2,
        metric_name="auroc",
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
    )


def bootstrap_delta_tpr_at_fpr(metrics1, metrics2, n_bootstrap=1000, random_seed=42):
    return bootstrap_metric_delta(
        metrics1,
        metrics2,
        metric_name="tpr_at_fpr",
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
    )


def required_sample_size_for_delta(
    metrics1,
    metrics2,
    w_target=0.02,
    metric_name="auroc",
    n_bootstrap=1000,
    alpha=0.05,
):
    delta_boot = bootstrap_metric_delta(
        metrics1,
        metrics2,
        metric_name=metric_name,
        n_bootstrap=n_bootstrap,
    )
    n_pilot = len(delta_boot)
    lower = np.nanpercentile(delta_boot, 100 * alpha / 2)
    upper = np.nanpercentile(delta_boot, 100 * (1 - alpha / 2))
    w_pilot = upper - lower
    n_target = n_pilot * (w_pilot / w_target) ** 2
    return int(np.ceil(n_target)), float(w_pilot)
