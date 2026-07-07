import os
from itertools import combinations
from os.path import join as j
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from utils.evaluation.Metrics import Metrics
from utils.evaluation.Predictions import Predictions


DEFAULT_REGIME_ORDER = ("human", "free_llm", "h2l", "llm2l")
POSITIVE_REGIME_PRIORITY = {
    "human": 0,
    "h2l": 1,
    "llm2l": 2,
    "free_llm": 3,
}


def get_human_evaluation_path(model_name, dataset_name, lang=""):
    """Generates a filename based on model name, dataset name, and language."""
    os.makedirs("results", exist_ok=True)
    if lang:
        return j("results", f"human_{model_name}_{dataset_name}_{lang}.npz")
    return j("results", f"human_{model_name}_{dataset_name}.npz")


def get_evaluation_path(model_name, dataset_name):
    """Generates a filename based on model name and dataset name."""
    os.makedirs("results", exist_ok=True)
    return j("results", f"{model_name}_{dataset_name}.npz")


def _ordered_regimes(regimes: Optional[Sequence[str]]):
    regimes = list(regimes or DEFAULT_REGIME_ORDER)
    return sorted(regimes, key=lambda regime: DEFAULT_REGIME_ORDER.index(regime) if regime in DEFAULT_REGIME_ORDER else len(DEFAULT_REGIME_ORDER))


def choose_positive_regime(regimes: Sequence[str]):
    """
    Keeps the positive class aligned with the detector's "machine-likeness" score.
    """
    return max(regimes, key=lambda regime: POSITIVE_REGIME_PRIORITY.get(regime, -1))


def build_predictions_for_comparison(
    frame: pd.DataFrame,
    positive_regimes: Sequence[str],
    negative_regimes: Sequence[str],
    score_col: str,
    regime_col: str = "regime",
    id_col: str = "sample_id",
):
    relevant_regimes = list(positive_regimes) + list(negative_regimes)
    subset = frame[frame[regime_col].isin(relevant_regimes)].copy()
    if subset.empty:
        raise ValueError("No rows found for the requested regimes.")

    true_labels = subset[regime_col].isin(positive_regimes).astype(int).tolist()
    pred_probs = subset[score_col].astype(float).tolist()
    predicted_labels = (subset[score_col].astype(float) >= 0.5).astype(int).tolist()

    preds = Predictions(
        predicted_labels=predicted_labels,
        true_labels=true_labels,
        pred_probs=pred_probs,
    )
    if id_col in subset.columns:
        preds.set_ids(subset[id_col].tolist())
    return preds


def evaluate_binary_comparison(
    frame: pd.DataFrame,
    positive_regimes: Sequence[str],
    negative_regimes: Sequence[str],
    score_col: str,
    regime_col: str = "regime",
    id_col: str = "sample_id",
    comparison_name: Optional[str] = None,
    comparison_type: str = "pairwise",
    tpr_fpr_target: float = 0.01,
    ece_bins: int = 10,
):
    preds = build_predictions_for_comparison(
        frame=frame,
        positive_regimes=positive_regimes,
        negative_regimes=negative_regimes,
        score_col=score_col,
        regime_col=regime_col,
        id_col=id_col,
    )

    metadata = {
        "comparison_type": comparison_type,
        "positive_regimes": list(positive_regimes),
        "negative_regimes": list(negative_regimes),
    }

    return Metrics(
        preds,
        tpr_fpr_target=tpr_fpr_target,
        ece_bins=ece_bins,
        comparison_name=comparison_name,
        metadata=metadata,
    )


def evaluate_regime_comparisons(
    frame: pd.DataFrame,
    score_col: str,
    regime_col: str = "regime",
    id_col: str = "sample_id",
    regimes: Optional[Sequence[str]] = None,
    include_one_vs_rest: bool = True,
    tpr_fpr_target: float = 0.01,
    ece_bins: int = 10,
):
    regimes = _ordered_regimes(regimes or frame[regime_col].dropna().unique().tolist())
    metrics: List[Metrics] = []

    for left, right in combinations(regimes, 2):
        positive = choose_positive_regime((left, right))
        negative = right if positive == left else left
        metrics.append(
            evaluate_binary_comparison(
                frame=frame,
                positive_regimes=[positive],
                negative_regimes=[negative],
                score_col=score_col,
                regime_col=regime_col,
                id_col=id_col,
                comparison_name=f"{positive}_vs_{negative}",
                comparison_type="pairwise",
                tpr_fpr_target=tpr_fpr_target,
                ece_bins=ece_bins,
            )
        )

    if include_one_vs_rest:
        for positive in regimes:
            negative_regimes = [regime for regime in regimes if regime != positive]
            metrics.append(
                evaluate_binary_comparison(
                    frame=frame,
                    positive_regimes=[positive],
                    negative_regimes=negative_regimes,
                    score_col=score_col,
                    regime_col=regime_col,
                    id_col=id_col,
                    comparison_name=f"{positive}_vs_rest",
                    comparison_type="one_vs_rest",
                    tpr_fpr_target=tpr_fpr_target,
                    ece_bins=ece_bins,
                )
            )

    return metrics


def metrics_to_frame(metrics: Iterable[Metrics]):
    return Metrics.to_frame(list(metrics))


def compute_rewrite_deltas(
    frame: pd.DataFrame,
    score_col: str,
    regime_col: str = "regime",
    id_col: str = "sample_id",
    tpr_fpr_target: float = 0.01,
    ece_bins: int = 10,
):
    """
    Measures the effect of rewriting while keeping the opposite side of the comparison fixed.

    FREE-LLM -> LLM2L:
        compare HUMAN vs FREE-LLM against HUMAN vs LLM2L

    HUMAN -> H2L:
        compare FREE-LLM vs HUMAN against FREE-LLM vs H2L
    """
    baseline_human_vs_free = evaluate_binary_comparison(
        frame=frame,
        positive_regimes=["free_llm"],
        negative_regimes=["human"],
        score_col=score_col,
        regime_col=regime_col,
        id_col=id_col,
        comparison_name="free_llm_vs_human",
        comparison_type="rewrite_anchor",
        tpr_fpr_target=tpr_fpr_target,
        ece_bins=ece_bins,
    )

    rewritten_llm = evaluate_binary_comparison(
        frame=frame,
        positive_regimes=["llm2l"],
        negative_regimes=["human"],
        score_col=score_col,
        regime_col=regime_col,
        id_col=id_col,
        comparison_name="llm2l_vs_human",
        comparison_type="rewrite_delta",
        tpr_fpr_target=tpr_fpr_target,
        ece_bins=ece_bins,
    )

    rewritten_human = evaluate_binary_comparison(
        frame=frame,
        positive_regimes=["free_llm"],
        negative_regimes=["h2l"],
        score_col=score_col,
        regime_col=regime_col,
        id_col=id_col,
        comparison_name="free_llm_vs_h2l",
        comparison_type="rewrite_delta",
        tpr_fpr_target=tpr_fpr_target,
        ece_bins=ece_bins,
    )

    rows = [
        {
            "delta_name": "free_llm_to_llm2l",
            "baseline_comparison": baseline_human_vs_free.comparison_name,
            "target_comparison": rewritten_llm.comparison_name,
            "delta_auroc": rewritten_llm.auroc - baseline_human_vs_free.auroc,
            "delta_tpr_at_1pct_fpr": rewritten_llm.tpr_at_1pct_fpr - baseline_human_vs_free.tpr_at_1pct_fpr,
        },
        {
            "delta_name": "human_to_h2l",
            "baseline_comparison": baseline_human_vs_free.comparison_name,
            "target_comparison": rewritten_human.comparison_name,
            "delta_auroc": rewritten_human.auroc - baseline_human_vs_free.auroc,
            "delta_tpr_at_1pct_fpr": rewritten_human.tpr_at_1pct_fpr - baseline_human_vs_free.tpr_at_1pct_fpr,
        },
    ]

    return pd.DataFrame(rows)
