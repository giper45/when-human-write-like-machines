from typing import List

import pandas as pd
import pandera as pa

from utils.evaluation.BootstrapMetrics import bootstrap_metrics_for_detector, bootstrap_paired_delta_for_detector, summarize_bootstrap_array
from utils.evaluation.Metrics import Metrics, align_metric_blocks, filter_metrics_by_target_regime

# Define the schema as a Pandera DataFrameSchema object
DetectorTableSchema = pa.DataFrameSchema(
    columns={
        # Use the MultiIndex tuple as the column key
        ("AUROC", "mean ± 95% CI"): pa.Column(str, coerce=True),
        ("TPR@1%FPR", "mean ± 95% CI"): pa.Column(str, coerce=True),
    },
    # Enforce index name validation
    index=pa.Index(str, name="Detector"),
    strict=True
)

H2LRobustnessTableSchema = pa.DataFrameSchema(
    columns={
        ("AUROC H2L", "mean [95% CI]"): pa.Column(str, coerce=True),
        ("ΔAUROC H2L", "mean [95% CI]"): pa.Column(str, coerce=True),
        ("TPR@1%FPR H2L", "mean [95% CI]"): pa.Column(str, coerce=True),
        ("ΔTPR@1%FPR H2L", "mean [95% CI]"): pa.Column(str, coerce=True),
    },
    index=pa.Index(str, name="Detector"),
    strict=True,
)

# Example usage for runtime validation (and static type checking friendliness)
def validate (df: pd.DataFrame) -> pd.DataFrame:
    # Validate the DataFrame against the schema
    validated_df = DetectorTableSchema.validate(df)
    return validated_df

def validate_h2l_robustness_table(df: pd.DataFrame) -> pd.DataFrame:
    return H2LRobustnessTableSchema.validate(df)

def format_mean_ci_from_summary(summary, metric_prefix, digits=3):
    mean = summary[f"{metric_prefix}_mean"]
    low = summary[f"{metric_prefix}_ci_low"]
    high = summary[f"{metric_prefix}_ci_high"]

    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


import pandas as pd

def format_summary_ci(summary, digits=3):
    return (
        f"{summary['mean']:.{digits}f} "
        f"[{summary['ci_low']:.{digits}f}, {summary['ci_high']:.{digits}f}]"
    )


def build_detector_table(
    results_dir,
    detector_names,
    baseline_regime="free_llm",
    n_bootstrap=5000,
    random_seed=42,
    alpha=0.05,
    digits=3,
):
    rows = []

    for detector_name in detector_names:
        all_metrics = Metrics.load_metrics_of_detector(
            results_dir,
            detector_name,
        )
        baseline_metrics = filter_metrics_by_target_regime(
            all_metrics,
            baseline_regime,
        )


        bm = bootstrap_metrics_for_detector(
            baseline_metrics,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )

        s = bm.summary(alpha=alpha)

        rows.append({
            "Detector": detector_name,
            ("AUROC FREE-LLM", "mean ± 95% CI"): format_mean_ci_from_summary(
                s,
                "auroc",
                digits=digits,
            ),
            ("TPR@1%FPR", "mean ± 95% CI"): format_mean_ci_from_summary(
                s,
                "tpr_at_fpr",
                digits=digits,
            ),
        })

    df = pd.DataFrame(rows)
    df = df.set_index("Detector")
    df.index.name = "Detector"

    df.columns = pd.MultiIndex.from_tuples(df.columns)

    return df

    # Convert the metrics dictionary to a DataFrame
    # print(metrics)
    # df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # # Validate the DataFrame against the schema
    # validated_df = validate(df)
    
    # return validated_df

def _build_from_to_robustness_table(
    results_dir,
    detector_names,
    source_regime,
    target_regime,
    target_regime_name,
    n_bootstrap=5000,
    random_seed=42,
    alpha=0.05,
    digits=3,
):
    rows = []

    for detector_name in detector_names:
        all_metrics = Metrics.load_metrics_of_detector(
            results_dir,
            detector_name,
        )

        baseline_metrics = filter_metrics_by_target_regime(
            all_metrics,
            source_regime,
        )

        h2l_metrics = filter_metrics_by_target_regime(
            all_metrics,
            target_regime,
        )

        baseline_metrics, h2l_metrics = align_metric_blocks(
            baseline_metrics,
            h2l_metrics,
        )

        h2l_bm = bootstrap_metrics_for_detector(
            h2l_metrics,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )

        h2l_summary = h2l_bm.summary(alpha=alpha)

        delta_auroc_array = bootstrap_paired_delta_for_detector(
            baseline_metrics=baseline_metrics,
            target_metrics=h2l_metrics,
            metric_name="auroc",
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )

        delta_tpr_array = bootstrap_paired_delta_for_detector(
            baseline_metrics=baseline_metrics,
            target_metrics=h2l_metrics,
            metric_name="tpr_at_fpr",
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )

        delta_auroc_summary = summarize_bootstrap_array(
            delta_auroc_array,
            alpha=alpha,
        )

        delta_tpr_summary = summarize_bootstrap_array(
            delta_tpr_array,
            alpha=alpha,
        )

        rows.append({
            "Detector": detector_name,

            ("AUROC " + target_regime_name, "mean [95% CI]"): format_mean_ci_from_summary(
                h2l_summary,
                "auroc",
                digits=digits,
            ),

            ("ΔAUROC " + target_regime_name, "mean [95% CI]"): format_summary_ci(
                delta_auroc_summary,
                digits=digits,
            ),

            ("TPR@1%FPR " + target_regime_name, "mean [95% CI]"): format_mean_ci_from_summary(
                h2l_summary,
                "tpr_at_fpr",
                digits=digits,
            ),

            ("ΔTPR@1%FPR " + target_regime_name, "mean [95% CI]"): format_summary_ci(
                delta_tpr_summary,
                digits=digits,
            ),
        })

    df = pd.DataFrame(rows)
    df = df.set_index("Detector")
    df.index.name = "Detector"
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    return df



def build_llm2l_robustness_table(
    results_dir,
    detector_names,
    n_bootstrap=5000,
    random_seed=42,
    alpha=0.05,
    digits=3,
):
    return _build_from_to_robustness_table(
        results_dir=results_dir,
        detector_names=detector_names,
        source_regime="free_llm",
        target_regime="llm2l",
        target_regime_name="LLM2L",
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
        alpha=alpha,
        digits=digits,
    )
def build_h2l_robustness_table(
    results_dir,
    detector_names,
    n_bootstrap=5000,
    random_seed=42,
    alpha=0.05,
    digits=3,
):
    return _build_from_to_robustness_table(
        results_dir=results_dir,
        detector_names=detector_names,
        source_regime="free_llm",
        target_regime="h2l",
        target_regime_name="H2L",
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
        alpha=alpha,
        digits=digits,
    )

