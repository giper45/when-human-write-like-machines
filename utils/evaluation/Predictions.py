import numpy as np
from typing import Optional


VALID_SCORE_DIRECTIONS = {"higher_is_ai", "lower_is_ai"}


class Predictions:
    def __init__(
        self,
        predicted_labels: list,
        true_labels: list,
        pred_probs: Optional[list] = None,
        negate: bool = False,
        *,
        scores: Optional[list] = None,
        raw_scores: Optional[list] = None,
        default_threshold: Optional[float] = 0.5,
        score_direction: str = "higher_is_ai",
        ids: Optional[list] = None,
        metadata: Optional[dict] = None,
        scores_are_probabilities: Optional[bool] = None,
    ):
        if score_direction not in VALID_SCORE_DIRECTIONS:
            raise ValueError(
                f"score_direction must be one of {sorted(VALID_SCORE_DIRECTIONS)}"
            )

        if pred_probs is None and scores is None:
            raise ValueError("Predictions requires either scores or legacy pred_probs.")

        if scores is None:
            scores = pred_probs
        if raw_scores is None:
            raw_scores = scores

        if negate:
            print("Negating predictions (for inverse-label detector predictions)")
            predicted_labels = [1 - label for label in predicted_labels]
            scores = [1 - score for score in scores]
            if pred_probs is not None:
                pred_probs = [1 - prob for prob in pred_probs]
            raw_scores = scores

        self.predicted_labels = list(predicted_labels)
        self.true_labels = list(true_labels)
        self.scores = list(scores)
        self.raw_scores = list(raw_scores)
        self.default_threshold = default_threshold
        self.score_direction = score_direction
        self.metadata = metadata or {}

        if scores_are_probabilities is None:
            scores_are_probabilities = self._are_valid_probabilities(self.scores)
        self.scores_are_probabilities = bool(scores_are_probabilities)

        # Backward compatibility: older notebooks/scripts read pred_probs. New code
        # should prefer scores and check scores_are_probabilities before calibration
        # metrics.
        self.pred_probs = list(pred_probs) if pred_probs is not None else list(self.scores)

        self.error_probs = []
        self.error_preds = []
        self.error_threshold = None
        self.is_error_based_classifier = False
        self.ids = list(ids) if ids is not None else []

    @staticmethod
    def _are_valid_probabilities(values):
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return False
        return bool(np.all(np.isfinite(values)) and np.all((values >= 0.0) & (values <= 1.0)))


    def set_ids(self, ids):
        self.ids = list(ids)

    def set_errors(self, error_probs, error_preds, error_threshold):
        self.error_probs = error_probs
        self.error_preds = error_preds
        self.error_threshold = error_threshold
        self.is_error_based_classifier = True
