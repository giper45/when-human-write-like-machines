from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from datasets import load_from_disk

from utils.evaluation.Metrics import Metrics, get_prediction_scores
from utils.evaluation.threshold_transfer import apply_threshold, select_threshold_at_fpr


REGIME_ALIASES = {"freellm": "free_llm", "llmfree": "free_llm"}
REGIME_ORDER = {"human": 0, "free_llm": 1, "h2l": 2, "llm2l": 3}
REWRITE_REGIMES = {"h2l", "llm2l"}


def _normalize_regime(regime: Any) -> str:
    value = str(regime)
    return REGIME_ALIASES.get(value, value)


class ScoreAnalyzer:
    """Build a sample-level, long-form score table from detector metrics.

    Prediction IDs written by run_detector retain the source index (for
    example "human::12" and "h2l::12"). The index is used to join scores
    back to the sampled Hugging Face dataset and generated-text file.

    With multiple metrics, repeated human rows produced by separate regime
    runs are collapsed in each dataset/generator/detector block.
    """

    COLUMNS = [
        "sample_id",
        "dataset",
        "length_band",
        "generator",
        "rewriter",
        "regime",
        "text",
        "detector",
        "score",
        "threshold",
        "label",
        "predicted_label",
    ]

    def __init__(
        self,
        metrics: Metrics | Iterable[Metrics],
    ) -> None:
        self.metrics = [metrics] if isinstance(metrics, Metrics) else list(metrics)
        self._dataset_cache: dict[Path, Any] = {}
        self._machine_text_cache: dict[Path, list[str]] = {}

    def to_dataframe(self) -> pd.DataFrame:
        """Return one row per sample, regime and detector score."""
        rows: list[dict[str, Any]] = []
        for metric_index, metric in enumerate(self.metrics):
            rows.extend(self._metric_rows(metric, metric_index))

        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(columns=self.COLUMNS)

        # Human texts are evaluated again in every target-regime run, but are
        # logically one observation in a detector/generator block.
        identity_columns = [
            "_source_index",
            "dataset",
            "generator",
            "rewriter",
            "regime",
            "detector",
        ]
        frame = frame.drop_duplicates(subset=identity_columns, keep="first")

        frame["_regime_order"] = frame["regime"].map(REGIME_ORDER).fillna(99)
        frame = frame.sort_values(
            [
                "dataset",
                "generator",
                "detector",
                "_source_index",
                "_regime_order",
                "_metric_index",
            ],
            kind="stable",
        )
        return frame.loc[:, self.COLUMNS].reset_index(drop=True)

    # Aliases useful in notebooks and action-oriented callers.
    def build_dataframe(self) -> pd.DataFrame:
        return self.to_dataframe()

    def analyze(self) -> pd.DataFrame:
        return self.to_dataframe()

    def _metric_rows(self, metric: Metrics, metric_index: int) -> list[dict[str, Any]]:
        metadata = metric.metadata or {}
        dataset_name = self._required_metadata(metadata, "dataset_name")
        generator_name = self._required_metadata(metadata, "generator_name")
        detector_name = self._required_metadata(metadata, "detector_name")
        target_regime = _normalize_regime(
            metadata.get("target_regime")
            or metadata.get("machine_postfix")
            or self._target_from_comparison(metric.comparison_name)
        )

        sampled = self._load_sampled_dataset(metadata)
        machine_texts = self._load_machine_texts(metadata)

        labels = list(metric.preds.true_labels)
        scores = list(get_prediction_scores(metric.preds))
        ids = list(getattr(metric.preds, "ids", []) or [])
        if len(labels) != len(scores):
            raise ValueError(
                f"Metric {detector_name}/{dataset_name}/{target_regime} has "
                f"{len(labels)} labels but {len(scores)} scores."
            )
        if ids and len(ids) != len(scores):
            raise ValueError(
                f"Metric {detector_name}/{dataset_name}/{target_regime} has "
                f"{len(ids)} IDs but {len(scores)} scores."
            )

        threshold, _ = select_threshold_at_fpr(
            labels,
            scores,
            target_fpr=metric.tpr_fpr_target,
        )
        predicted_labels = apply_threshold(scores, threshold).tolist()

        rows = []
        label_offsets = {0: 0, 1: 0}
        for position, (label, predicted_label, score) in enumerate(
            zip(labels, predicted_labels, scores)
        ):
            numeric_label = int(label)
            if numeric_label not in label_offsets:
                raise ValueError(f"Only binary labels 0/1 are supported; found {label!r}.")

            prediction_id = ids[position] if ids else None
            regime, source_index, explicit_sample_id = self._prediction_reference(
                prediction_id,
                numeric_label,
                target_regime,
                label_offsets[numeric_label],
            )
            label_offsets[numeric_label] += 1

            expected_label = 0 if regime == "human" else 1
            if numeric_label != expected_label:
                raise ValueError(
                    f"Prediction ID {prediction_id!r} implies regime {regime!r}, "
                    f"but its true label is {numeric_label}."
                )
            if regime not in {"human", target_regime}:
                raise ValueError(
                    f"Prediction ID {prediction_id!r} has regime {regime!r}; "
                    f"expected 'human' or {target_regime!r}."
                )

            try:
                sample = sampled[source_index]
            except (IndexError, KeyError) as exc:
                raise IndexError(
                    f"Source index {source_index} from prediction ID "
                    f"{prediction_id!r} is outside sampled dataset "
                    f"{dataset_name!r} (size {len(sampled)})."
                ) from exc

            if regime == "human":
                text = sample["text"]
            else:
                try:
                    text = machine_texts[source_index]
                except IndexError as exc:
                    raise IndexError(
                        f"Source index {source_index} from prediction ID "
                        f"{prediction_id!r} is outside the machine-text file "
                        f"(size {len(machine_texts)})."
                    ) from exc

            sample_id = explicit_sample_id
            if sample_id is None:
                sample_id = sample.get("sample_id", sample.get("id", source_index))

            rows.append(
                {
                    "sample_id": sample_id,
                    "dataset": dataset_name,
                    "length_band": sample.get(
                        "length_band", sample.get("range", metadata.get("length_band"))
                    ),
                    "generator": generator_name,
                    "rewriter": self._rewriter(metadata, regime, generator_name),
                    "regime": regime,
                    "text": text,
                    "detector": detector_name,
                    "score": float(score),
                    "threshold": float(threshold),
                    "label": numeric_label,
                    "predicted_label": int(predicted_label),
                    "_source_index": source_index,
                    "_metric_index": metric_index,
                }
            )
        return rows

    @staticmethod
    def _required_metadata(metadata: Mapping[str, Any], key: str) -> Any:
        value = metadata.get(key)
        if value is None or value == "":
            raise ValueError(f"Metrics metadata is missing required key {key!r}.")
        return value

    @staticmethod
    def _target_from_comparison(comparison_name: Optional[str]) -> str:
        if comparison_name and "_vs_" in comparison_name:
            return comparison_name.split("_vs_", 1)[0]
        raise ValueError(
            "Metrics metadata must include 'target_regime' (or 'machine_postfix')."
        )

    @staticmethod
    def _prediction_reference(
        prediction_id: Any,
        label: int,
        target_regime: str,
        fallback_index: int,
    ) -> tuple[str, int, Any]:
        fallback_regime = "human" if label == 0 else target_regime
        if prediction_id is None:
            return fallback_regime, fallback_index, None

        if isinstance(prediction_id, str) and "::" in prediction_id:
            regime, raw_index = prediction_id.rsplit("::", 1)
            try:
                source_index = int(raw_index)
            except ValueError as exc:
                raise ValueError(
                    f"Prediction ID {prediction_id!r} must end in an integer source index."
                ) from exc
            if source_index < 0:
                raise ValueError(f"Prediction ID {prediction_id!r} has a negative index.")
            return _normalize_regime(regime), source_index, None

        # Generic Metrics may carry the logical sample ID rather than a
        # run_detector source reference. Class-relative position still locates
        # its source text.
        return fallback_regime, fallback_index, prediction_id

    def _load_sampled_dataset(self, metadata: Mapping[str, Any]):
        raw_path = metadata.get("sampled_dataset_path")
        if not raw_path:
            raise ValueError("Metrics metadata is missing 'sampled_dataset_path'.")
        path = Path(raw_path).resolve()
        if path not in self._dataset_cache:
            if not path.exists():
                raise FileNotFoundError(f"Sampled dataset not found: {path}.")
            self._dataset_cache[path] = load_from_disk(str(path))

        sampled = self._dataset_cache[path]
        if "text" not in sampled.column_names:
            raise ValueError(f"Sampled dataset {path} has no 'text' column.")
        return sampled

    def _load_machine_texts(self, metadata: Mapping[str, Any]) -> list[str]:
        raw_path = metadata.get("machine_texts_path")
        if not raw_path:
            raise ValueError("Metrics metadata is missing 'machine_texts_path'.")
        path = Path(raw_path).resolve()
        if path not in self._machine_text_cache:
            if not path.exists():
                raise FileNotFoundError(f"Machine-text file not found: {path}.")
            with path.open("r", encoding="utf-8") as handle:
                self._machine_text_cache[path] = [line.rstrip("\n") for line in handle]
        return self._machine_text_cache[path]

    @staticmethod
    def _rewriter(
        metadata: Mapping[str, Any], regime: str, generator_name: Any
    ) -> Any:
        if regime not in REWRITE_REGIMES:
            return None
        return metadata.get("rewriter_name") or metadata.get("rewriter") or generator_name
