import json
import os
from typing import Dict, List, Optional
from utils.logger import log

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from utils.evaluation.Predictions import Predictions


def drop_deprecated_summary_keys(metadata):
    deprecated_keys = {"ece", "brier", "ece_bins", "delta_ece", "delta_brier"}
    return {
        key: value
        for key, value in (metadata or {}).items()
        if key not in deprecated_keys and not str(key).endswith("_1pct_fpr")
    }


def specificity_score(y_true, y_pred):
    """Calculate specificity (True Negative Rate)."""
    cm = confusion_matrix(y_true, y_pred)
    return cm[0, 0] / (cm[0, 0] + cm[0, 1])


def scores_are_valid_probabilities(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return False
    return bool(np.all(np.isfinite(scores)) and np.all((scores >= 0.0) & (scores <= 1.0)))


def get_prediction_scores(preds: Predictions):
    return getattr(preds, "scores", getattr(preds, "pred_probs", []))


def compute_tpr_at_fpr(y_true, scores, target_fpr=0.01):
    """
    Returns the best achievable TPR while keeping FPR below the requested cap.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    try:
        fpr, tpr, _ = roc_curve(y_true, scores)
    except ValueError:
        return np.nan

    valid = fpr <= target_fpr
    if not np.any(valid):
        return 0.0
    return float(np.max(tpr[valid]))


def compute_ece(y_true, scores, n_bins=10):
    """
    Expected Calibration Error for binary probabilistic predictions.
    """
    y_true = np.asarray(y_true, dtype=float)
    scores = np.asarray(scores, dtype=float)

    if y_true.size == 0:
        return np.nan
    if not scores_are_valid_probabilities(scores):
        return np.nan

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for idx in range(n_bins):
        lower = bin_edges[idx]
        upper = bin_edges[idx + 1]
        if idx == n_bins - 1:
            mask = (scores >= lower) & (scores <= upper)
        else:
            mask = (scores >= lower) & (scores < upper)

        if not np.any(mask):
            continue

        bin_accuracy = y_true[mask].mean()
        bin_confidence = scores[mask].mean()
        ece += np.abs(bin_accuracy - bin_confidence) * mask.mean()

    return float(ece)


def compute_brier_score(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if y_true.size == 0:
        return np.nan
    if not scores_are_valid_probabilities(scores):
        return np.nan
    return float(brier_score_loss(y_true, scores))


def compute_auroc(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return np.nan


class Metrics:
    def __init__(
        self,
        preds: Predictions,
        tpr_fpr_target: float = 0.01,
        ece_bins: int = 10,
        comparison_name: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        self.preds = preds
        self.tpr_fpr_target = tpr_fpr_target
        self.ece_bins = ece_bins
        self.comparison_name = comparison_name
        self.metadata = drop_deprecated_summary_keys(metadata)
        self.scores = np.asarray(get_prediction_scores(self.preds), dtype=float)
        self.scores_are_probabilities = bool(
            getattr(self.preds, "scores_are_probabilities", False)
        ) and scores_are_valid_probabilities(self.scores)

        self.auroc = self._safe_auroc()
        self.tpr_at_fpr = compute_tpr_at_fpr(
            self.preds.true_labels,
            self.scores,
            target_fpr=self.tpr_fpr_target,
        )

        # Legacy fields kept for compatibility with earlier notebooks/scripts.
        self.f1 = f1_score(
            self.preds.true_labels,
            self.preds.predicted_labels,
            zero_division=0,
        )
        self.precision = precision_score(
            self.preds.true_labels,
            self.preds.predicted_labels,
            zero_division=0,
        )
        self.recall = recall_score(
            self.preds.true_labels,
            self.preds.predicted_labels,
            zero_division=0,
        )
        self.accuracy = accuracy_score(self.preds.true_labels, self.preds.predicted_labels)
        self.balanced_accuracy = balanced_accuracy_score(
            self.preds.true_labels,
            self.preds.predicted_labels,
        )

    def _safe_auroc(self):
        return compute_auroc(self.preds.true_labels, self.scores)

    def __str__(self):
        return (
            f"AUROC: {self.auroc:.4f}, "
            f"TPR@{self.tpr_fpr_target:.0%}FPR: {self.tpr_at_fpr:.4f}"
        )

    def __repr__(self):
        return (
            "Metrics("
            f"auroc={self.auroc:.4f}, "
            f"tpr_at_fpr={self.tpr_at_fpr:.4f})"
        )

    def summary(self):
        summary = {
            "comparison_name": self.comparison_name,
            "auroc": self.auroc,
            "tpr_at_fpr": self.tpr_at_fpr,
            "scores_are_probabilities": self.scores_are_probabilities,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "tpr_fpr_target": self.tpr_fpr_target,
        }
        summary.update(self.metadata)
        return summary

    def delta(self, baseline: "Metrics"):
        return {
            "delta_auroc": self.auroc - baseline.auroc,
            "delta_tpr_at_fpr": self.tpr_at_fpr - baseline.tpr_at_fpr,
        }

    def get_classification_indices(self):
        """
        Returns the indices of True Positives, False Positives, True Negatives, and False Negatives.
        """
        true_labels = np.array(self.preds.true_labels)
        predicted_labels = np.array(self.preds.predicted_labels)

        tp_indices = np.where((true_labels == 1) & (predicted_labels == 1))[0].tolist()
        fp_indices = np.where((true_labels == 0) & (predicted_labels == 1))[0].tolist()
        tn_indices = np.where((true_labels == 0) & (predicted_labels == 0))[0].tolist()
        fn_indices = np.where((true_labels == 1) & (predicted_labels == 0))[0].tolist()

        return {
            "tp": tp_indices,
            "fp": fp_indices,
            "tn": tn_indices,
            "fn": fn_indices,
        }

    def get_thresholded_classification_indices(self, threshold=0.05):
        """
        Returns classification indices based on a custom probability threshold.
        """
        pred_probs = np.array(get_prediction_scores(self.preds))
        true_labels = np.array(self.preds.true_labels)

        predicted_labels = (pred_probs >= threshold).astype(int)
        tp_indices = np.where((true_labels == 1) & (predicted_labels == 1))[0].tolist()
        fp_indices = np.where((true_labels == 0) & (predicted_labels == 1))[0].tolist()
        tn_indices = np.where((true_labels == 0) & (predicted_labels == 0))[0].tolist()
        fn_indices = np.where((true_labels == 1) & (predicted_labels == 0))[0].tolist()

        return {
            "tp": tp_indices,
            "fp": fp_indices,
            "tn": tn_indices,
            "fn": fn_indices,
            "pred_labels": predicted_labels.tolist(),
            "pred_probs": pred_probs.tolist(),
            "true_labels": true_labels.tolist(),
            "ids": getattr(self.preds, "ids", []),
        }

    def compute_thresholded_metrics(self, threshold=0.5):
        """
        Legacy helper retained for backward compatibility with older scripts.
        """
        thresholded_preds = [1 if score >= threshold else 0 for score in get_prediction_scores(self.preds)]
        self.f1 = f1_score(self.preds.true_labels, thresholded_preds)
        self.recall = recall_score(self.preds.true_labels, thresholded_preds)
        self.auroc = self._safe_auroc()

    def compute_error_based_thresholded_metrics(self, threshold=0.5):
        """
        Legacy helper retained for backward compatibility with older scripts.
        """
        thresholded_preds = [1 if score >= threshold else 0 for score in get_prediction_scores(self.preds)]
        for i, error_prob in enumerate(self.preds.error_probs):
            if error_prob >= self.preds.error_threshold:
                thresholded_preds[i] = 1 - thresholded_preds[i]
        self.f1 = f1_score(self.preds.true_labels, thresholded_preds)
        self.recall = recall_score(self.preds.true_labels, thresholded_preds)
        self.auroc = self._safe_auroc()

    def compute_error_based_thresholded_metrics_with_gating(self, threshold=0.5):
        """
        Legacy helper retained for backward compatibility with older scripts.
        """
        base_probs = np.array(get_prediction_scores(self.preds))
        final_preds = (base_probs >= threshold).astype(int)

        err = np.array(self.preds.error_probs)
        tau = self.preds.error_threshold
        kappa = getattr(self, "base_conf_threshold", 0.8)
        low_conf = np.minimum(base_probs, 1 - base_probs) >= (1 - kappa)

        flip_mask = (err >= tau) & low_conf
        final_preds[flip_mask] = 1 - final_preds[flip_mask]

        self.f1 = f1_score(self.preds.true_labels, final_preds)
        self.recall = recall_score(self.preds.true_labels, final_preds)
        self.auroc = self._safe_auroc()

    @classmethod
    def from_arrays(
        cls,
        true_labels,
        pred_probs=None,
        predicted_labels=None,
        ids=None,
        scores=None,
        raw_scores=None,
        default_threshold=0.5,
        score_direction="higher_is_ai",
        scores_are_probabilities=None,
        **kwargs,
    ):
        if scores is None:
            scores = pred_probs
        scores = np.asarray(scores, dtype=float)
        if pred_probs is not None:
            pred_probs = np.asarray(pred_probs, dtype=float).tolist()
        if predicted_labels is None:
            predicted_labels = (scores >= default_threshold).astype(int).tolist()

        preds = Predictions(
            predicted_labels=predicted_labels,
            true_labels=np.asarray(true_labels, dtype=int).tolist(),
            pred_probs=pred_probs,
            scores=scores.tolist(),
            raw_scores=raw_scores,
            default_threshold=default_threshold,
            score_direction=score_direction,
            scores_are_probabilities=scores_are_probabilities,
        )
        if ids is not None:
            preds.set_ids(list(ids))

        return cls(preds, **kwargs)

    @classmethod
    def load_from_file(cls, file_path):
        """Load a Metrics object from an .npz file."""
        data = np.load(file_path, allow_pickle=True)

        true_labels = data["true_labels"].tolist()
        predicted_labels = data["predicted_labels"].tolist()
        pred_probs = data["pred_probs"].tolist() if "pred_probs" in data else None
        scores = data["scores"].tolist() if "scores" in data else pred_probs
        raw_scores = data["raw_scores"].tolist() if "raw_scores" in data else scores
        default_threshold = (
            data["default_threshold"].item() if "default_threshold" in data else 0.5
        )
        score_direction = (
            data["score_direction"].item() if "score_direction" in data else "higher_is_ai"
        )
        scores_are_probabilities = (
            bool(data["scores_are_probabilities"].item())
            if "scores_are_probabilities" in data
            else None
        )

        preds = Predictions(
            predicted_labels=predicted_labels,
            true_labels=true_labels,
            pred_probs=pred_probs,
            scores=scores,
            raw_scores=raw_scores,
            default_threshold=default_threshold,
            score_direction=score_direction,
            scores_are_probabilities=scores_are_probabilities,
        )
        if "ids" in data:
            preds.set_ids(data["ids"].tolist())

        metadata = {}
        comparison_name = None
        tpr_fpr_target = 0.01
        ece_bins = 10

        if "metadata" in data:
            metadata = json.loads(data["metadata"].item())
            comparison_name = metadata.get("comparison_name")
            tpr_fpr_target = metadata.get("tpr_fpr_target", tpr_fpr_target)
            ece_bins = metadata.get("ece_bins", ece_bins)

        return cls(
            preds,
            tpr_fpr_target=tpr_fpr_target,
            ece_bins=ece_bins,
            comparison_name=comparison_name,
            metadata=metadata,
        )

    def save_to_file(
        self,
        file_path,
        detector_key,
        dataset_name,
        generator_key,
        temperature,
    ):
        """Save the Metrics object to an .npz file."""
        metadata = {
            "comparison_name": self.comparison_name,
            "detector_name": detector_key,
            "generator_name": generator_key,
            "temperature": temperature,
            "dataset_name": dataset_name,
            "auroc": self.auroc,
            "tpr_at_fpr": self.tpr_at_fpr,
            "scores_are_probabilities": self.scores_are_probabilities,
            "tpr_fpr_target": self.tpr_fpr_target,
        }
        metadata.update(self.metadata)

        log.info(f"Saving metrics to {file_path} with metadata: {metadata}")
        np.savez(
            file_path,
            true_labels=self.preds.true_labels,
            predicted_labels=self.preds.predicted_labels,
            pred_probs=self.preds.pred_probs,
            scores=get_prediction_scores(self.preds),
            raw_scores=getattr(self.preds, "raw_scores", get_prediction_scores(self.preds)),
            default_threshold=getattr(self.preds, "default_threshold", 0.5),
            score_direction=getattr(self.preds, "score_direction", "higher_is_ai"),
            scores_are_probabilities=getattr(self.preds, "scores_are_probabilities", False),
            ids=getattr(self.preds, "ids", []),
            metadata=json.dumps(metadata),
        )

    @staticmethod
    def load_from_folder(folder):
        metrics: List["Metrics"] = []
        npz_files = [os.path.join(folder, name) for name in os.listdir(folder) if name.endswith(".npz")]
        for npz_file in npz_files:
            metrics.append(Metrics.load_from_file(npz_file))
        return metrics


    @staticmethod
    def load_metrics_of_detector(folder, detector_name) -> List["Metrics"]:
        metrics: List["Metrics"] = []
        npz_files = [os.path.join(folder, name) for name in os.listdir(folder) if name.endswith(".npz")]
        for npz_file in npz_files:
            metric = Metrics.load_from_file(npz_file)
            if metric.metadata.get("detector_name") == detector_name:
                metrics.append(metric)
        return metrics

    @staticmethod
    def print_table(metrics):
        df = Metrics.to_frame(metrics)
        print(
            df.to_string(
                index=False,
                float_format="{:,.4f}".format,
            )
        )

    @staticmethod
    def write_excel(metrics, filename):
        df = Metrics.to_frame(metrics)
        df.to_excel(filename, index=False)

    @staticmethod
    def to_frame(metrics):
        rows = []
        for metric in metrics:
            rows.append(metric.summary())
        return pd.DataFrame(rows)


def filter_metrics_by_target_regime(metrics: List[Metrics], target_regime: str) -> List[Metrics]:
    target_regime = target_regime.lower()

    return [
        m for m in metrics
        if str(m.metadata.get("target_regime", "")).lower() == target_regime
        or str(m.comparison_name).lower().startswith(target_regime)
    ]

def metric_block_key(m: Metrics):
    return (
        m.metadata.get("dataset_name"),
        m.metadata.get("generator_name"),
        m.metadata.get("detector_name"),
    )

def align_metric_blocks(
    baseline_metrics: List[Metrics],
    target_metrics: List[Metrics],
):
    baseline_by_key = {
        metric_block_key(m): m
        for m in baseline_metrics
    }

    target_by_key = {
        metric_block_key(m): m
        for m in target_metrics
    }

    common_keys = sorted(set(baseline_by_key) & set(target_by_key))

    if not common_keys:
        raise ValueError("No common dataset × generator blocks between baseline and target.")

    missing_target = sorted(set(baseline_by_key) - set(target_by_key))
    missing_baseline = sorted(set(target_by_key) - set(baseline_by_key))

    if missing_target:
        print(f"WARNING: target missing blocks: {missing_target}")

    if missing_baseline:
        print(f"WARNING: baseline missing blocks: {missing_baseline}")

    aligned_baseline = [baseline_by_key[k] for k in common_keys]
    aligned_target = [target_by_key[k] for k in common_keys]

    return aligned_baseline, aligned_target