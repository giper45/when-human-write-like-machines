import numpy as np
import pandas as pd


TRANSFER_TARGET_REGIMES = ("free_llm", "h2l", "llm2l")
TRANSFER_COLUMNS = [
    "source_regime_for_threshold",
    "target_regime",
    "threshold",
    "target_fpr",
    "achieved_fpr_calibration",
    "test_fpr",
    "test_tpr",
    "test_precision",
    "test_recall",
    "test_specificity",
    "test_balanced_accuracy",
    "test_accuracy",
    "n_positive",
    "n_human",
    "delta_tpr_vs_free_llm",
    "delta_fpr_vs_free_llm",
]


def _safe_divide(numerator, denominator):
    if denominator == 0:
        return np.nan
    return float(numerator / denominator)


def _quantile_higher(values, quantile):
    try:
        return np.quantile(values, quantile, method="higher")
    except TypeError:
        return np.quantile(values, quantile, interpolation="higher")


def assign_calibration_test_split(
    frame,
    id_col="sample_id",
    calibration_size=0.20,
    seed=42,
):
    """
    Adds a deterministic calibration/test split by sample ID.
    """
    if id_col not in frame.columns:
        raise ValueError(f"Missing split ID column: {id_col}")

    result = frame.copy()
    sample_ids = sorted(result[id_col].dropna().unique().tolist(), key=lambda value: str(value))
    n_ids = len(sample_ids)
    if n_ids == 0:
        result["split"] = "test"
        return result

    calibration_size = float(calibration_size)
    if not 0.0 <= calibration_size <= 1.0:
        raise ValueError("calibration_size must be in [0, 1].")

    n_calibration = int(round(n_ids * calibration_size))
    if 0.0 < calibration_size < 1.0 and n_ids > 1:
        n_calibration = min(max(n_calibration, 1), n_ids - 1)

    rng = np.random.default_rng(seed)
    shuffled_ids = rng.permutation(np.asarray(sample_ids, dtype=object))
    calibration_ids = set(shuffled_ids[:n_calibration].tolist())

    result["split"] = np.where(
        result[id_col].isin(calibration_ids),
        "calibration",
        "test",
    )
    return result


def select_threshold_at_fpr(y_true, scores, target_fpr=0.01, positive_label=1):
    """
    Selects a threshold from negative-class scores for a target false-positive rate.
    Scores must already be normalized so higher means more AI-like.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    if y_true.shape[0] != scores.shape[0]:
        raise ValueError("y_true and scores must have the same length.")
    if not 0.0 <= float(target_fpr) <= 1.0:
        raise ValueError("target_fpr must be in [0, 1].")

    finite_mask = np.isfinite(scores)
    negative_scores = scores[(y_true != positive_label) & finite_mask]
    if negative_scores.size == 0:
        raise ValueError("Cannot select threshold without negative-class scores.")

    if target_fpr == 0:
        threshold = np.nextafter(float(np.max(negative_scores)), np.inf)
    else:
        threshold = float(_quantile_higher(negative_scores, 1.0 - float(target_fpr)))

    achieved_fpr = float(np.mean(negative_scores >= threshold))
    return threshold, achieved_fpr


def apply_threshold(scores, threshold):
    scores = np.asarray(scores, dtype=float)
    return (scores >= threshold).astype(int)


def threshold_metrics(y_true, scores, threshold):
    y_true = np.asarray(y_true).astype(int)
    predictions = apply_threshold(scores, threshold)

    tp = int(np.sum((y_true == 1) & (predictions == 1)))
    fp = int(np.sum((y_true == 0) & (predictions == 1)))
    tn = int(np.sum((y_true == 0) & (predictions == 0)))
    fn = int(np.sum((y_true == 1) & (predictions == 0)))

    tpr = _safe_divide(tp, tp + fn)
    fpr = _safe_divide(fp, fp + tn)
    precision = _safe_divide(tp, tp + fp)
    recall = tpr
    specificity = _safe_divide(tn, tn + fp)
    if np.isnan(tpr) or np.isnan(specificity):
        balanced_accuracy = np.nan
    else:
        balanced_accuracy = float((tpr + specificity) / 2.0)
    accuracy = _safe_divide(tp + tn, tp + fp + tn + fn)

    return {
        "threshold": float(threshold),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "TPR": tpr,
        "FPR": fpr,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "accuracy": accuracy,
    }


def threshold_transfer_report(
    calibration_df,
    test_df,
    score_col,
    regime_col,
    id_col,
    target_fpr=0.01,
):
    """
    Selects a FREE-LLM vs HUMAN threshold on calibration rows and applies it to
    HUMAN vs each target regime in the test rows.
    """
    for frame_name, frame in {"calibration_df": calibration_df, "test_df": test_df}.items():
        missing_cols = {score_col, regime_col} - set(frame.columns)
        if missing_cols:
            raise ValueError(f"{frame_name} is missing required columns: {sorted(missing_cols)}")
        if id_col is not None and id_col not in frame.columns:
            raise ValueError(f"{frame_name} is missing ID column: {id_col}")

    calibration_subset = calibration_df[
        calibration_df[regime_col].isin(["human", "free_llm"])
    ].copy()
    if calibration_subset.empty:
        raise ValueError("Calibration split has no HUMAN/FREE-LLM rows.")
    if not (calibration_subset[regime_col] == "human").any():
        raise ValueError("Calibration split has no HUMAN rows.")
    if not (calibration_subset[regime_col] == "free_llm").any():
        raise ValueError("Calibration split has no FREE-LLM rows.")

    y_calibration = (calibration_subset[regime_col] == "free_llm").astype(int).to_numpy()
    calibration_scores = calibration_subset[score_col].astype(float).to_numpy()
    threshold, achieved_fpr_calibration = select_threshold_at_fpr(
        y_calibration,
        calibration_scores,
        target_fpr=target_fpr,
        positive_label=1,
    )

    rows = []
    for target_regime in TRANSFER_TARGET_REGIMES:
        target_subset = test_df[test_df[regime_col].isin(["human", target_regime])].copy()
        y_test = (target_subset[regime_col] == target_regime).astype(int).to_numpy()
        test_scores = target_subset[score_col].astype(float).to_numpy()
        metrics = threshold_metrics(y_test, test_scores, threshold)

        rows.append(
            {
                "source_regime_for_threshold": "free_llm",
                "target_regime": target_regime,
                "threshold": metrics["threshold"],
                "target_fpr": float(target_fpr),
                "achieved_fpr_calibration": achieved_fpr_calibration,
                "test_fpr": metrics["FPR"],
                "test_tpr": metrics["TPR"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
                "test_specificity": metrics["specificity"],
                "test_balanced_accuracy": metrics["balanced_accuracy"],
                "test_accuracy": metrics["accuracy"],
                "n_positive": int(np.sum(y_test == 1)),
                "n_human": int(np.sum(y_test == 0)),
            }
        )

    report = pd.DataFrame(rows)
    if report.empty:
        return pd.DataFrame(columns=TRANSFER_COLUMNS)

    baseline = report[report["target_regime"] == "free_llm"]
    if baseline.empty:
        report["delta_tpr_vs_free_llm"] = np.nan
        report["delta_fpr_vs_free_llm"] = np.nan
    else:
        baseline_tpr = float(baseline["test_tpr"].iloc[0])
        baseline_fpr = float(baseline["test_fpr"].iloc[0])
        report["delta_tpr_vs_free_llm"] = report["test_tpr"] - baseline_tpr
        report["delta_fpr_vs_free_llm"] = report["test_fpr"] - baseline_fpr

    return report[TRANSFER_COLUMNS]
