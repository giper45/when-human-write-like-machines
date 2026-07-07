import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from utils.evaluation.Predictions import Predictions


def specificity_score(y_true, y_pred):
    """Calculate specificity (True Negative Rate)."""
    cm = confusion_matrix(y_true, y_pred)
    return cm[0, 0] / (cm[0, 0] + cm[0, 1])


def compute_tpr_at_fpr(y_true, pred_probs, target_fpr=0.01):
    """
    Returns the best achievable TPR while keeping FPR below the requested cap.
    """
    y_true = np.asarray(y_true)
    pred_probs = np.asarray(pred_probs)

    try:
        fpr, tpr, _ = roc_curve(y_true, pred_probs)
    except ValueError:
        return np.nan

    valid = fpr <= target_fpr
    if not np.any(valid):
        return 0.0
    return float(np.max(tpr[valid]))


def compute_ece(y_true, pred_probs, n_bins=10):
    """
    Expected Calibration Error for binary probabilistic predictions.
    """
    y_true = np.asarray(y_true, dtype=float)
    pred_probs = np.asarray(pred_probs, dtype=float)

    if y_true.size == 0:
        return np.nan

    pred_probs = np.clip(pred_probs, 0.0, 1.0)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for idx in range(n_bins):
        lower = bin_edges[idx]
        upper = bin_edges[idx + 1]
        if idx == n_bins - 1:
            mask = (pred_probs >= lower) & (pred_probs <= upper)
        else:
            mask = (pred_probs >= lower) & (pred_probs < upper)

        if not np.any(mask):
            continue

        bin_accuracy = y_true[mask].mean()
        bin_confidence = pred_probs[mask].mean()
        ece += np.abs(bin_accuracy - bin_confidence) * mask.mean()

    return float(ece)


def compute_brier_score(y_true, pred_probs):
    y_true = np.asarray(y_true)
    pred_probs = np.asarray(pred_probs)
    if y_true.size == 0:
        return np.nan
    return float(brier_score_loss(y_true, pred_probs))


def compute_auroc(y_true, pred_probs):
    y_true = np.asarray(y_true)
    pred_probs = np.asarray(pred_probs)
    try:
        return float(roc_auc_score(y_true, pred_probs))
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
        self.metadata = metadata or {}

        self.auroc = self._safe_auroc()
        self.tpr_at_fpr = compute_tpr_at_fpr(
            self.preds.true_labels,
            self.preds.pred_probs,
            target_fpr=self.tpr_fpr_target,
        )
        self.tpr_at_1pct_fpr = compute_tpr_at_fpr(
            self.preds.true_labels,
            self.preds.pred_probs,
            target_fpr=0.01,
        )
        self.ece = compute_ece(
            self.preds.true_labels,
            self.preds.pred_probs,
            n_bins=self.ece_bins,
        )
        self.brier = compute_brier_score(self.preds.true_labels, self.preds.pred_probs)

        # Legacy fields kept for compatibility with earlier notebooks/scripts.
        self.f1 = f1_score(self.preds.true_labels, self.preds.predicted_labels)
        self.recall = recall_score(self.preds.true_labels, self.preds.predicted_labels)

    def _safe_auroc(self):
        return compute_auroc(self.preds.true_labels, self.preds.pred_probs)

    def __str__(self):
        return (
            f"AUROC: {self.auroc:.4f}, "
            f"TPR@{self.tpr_fpr_target:.0%}FPR: {self.tpr_at_fpr:.4f}, "
            f"ECE: {self.ece:.4f}, "
            f"Brier: {self.brier:.4f}"
        )

    def __repr__(self):
        return (
            "Metrics("
            f"auroc={self.auroc:.4f}, "
            f"tpr_at_fpr={self.tpr_at_fpr:.4f}, "
            f"ece={self.ece:.4f}, "
            f"brier={self.brier:.4f})"
        )

    def summary(self):
        summary = {
            "comparison_name": self.comparison_name,
            "auroc": self.auroc,
            "tpr_at_fpr": self.tpr_at_fpr,
            "tpr_at_1pct_fpr": self.tpr_at_1pct_fpr,
            "ece": self.ece,
            "brier": self.brier,
            "tpr_fpr_target": self.tpr_fpr_target,
            "ece_bins": self.ece_bins,
        }
        summary.update(self.metadata)
        return summary

    def delta(self, baseline: "Metrics"):
        return {
            "delta_auroc": self.auroc - baseline.auroc,
            "delta_tpr_at_fpr": self.tpr_at_fpr - baseline.tpr_at_fpr,
            "delta_tpr_at_1pct_fpr": self.tpr_at_1pct_fpr - baseline.tpr_at_1pct_fpr,
            "delta_ece": self.ece - baseline.ece,
            "delta_brier": self.brier - baseline.brier,
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
        pred_probs = np.array(self.preds.pred_probs)
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
        thresholded_preds = [1 if prob >= threshold else 0 for prob in self.preds.pred_probs]
        self.f1 = f1_score(self.preds.true_labels, thresholded_preds)
        self.recall = recall_score(self.preds.true_labels, thresholded_preds)
        self.auroc = self._safe_auroc()

    def compute_error_based_thresholded_metrics(self, threshold=0.5):
        """
        Legacy helper retained for backward compatibility with older scripts.
        """
        thresholded_preds = [1 if prob >= threshold else 0 for prob in self.preds.pred_probs]
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
        base_probs = np.array(self.preds.pred_probs)
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
        pred_probs,
        predicted_labels=None,
        ids=None,
        **kwargs,
    ):
        pred_probs = np.asarray(pred_probs, dtype=float)
        if predicted_labels is None:
            predicted_labels = (pred_probs >= 0.5).astype(int).tolist()

        preds = Predictions(
            predicted_labels=predicted_labels,
            true_labels=np.asarray(true_labels, dtype=int).tolist(),
            pred_probs=pred_probs.tolist(),
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
        pred_probs = data["pred_probs"].tolist()

        preds = Predictions(predicted_labels, true_labels, pred_probs)
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
            "tpr_at_1pct_fpr": self.tpr_at_1pct_fpr,
            "ece": self.ece,
            "brier": self.brier,
            "tpr_fpr_target": self.tpr_fpr_target,
            "ece_bins": self.ece_bins,
        }
        metadata.update(self.metadata)

        np.savez(
            file_path,
            true_labels=self.preds.true_labels,
            predicted_labels=self.preds.predicted_labels,
            pred_probs=self.preds.pred_probs,
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
