import re

import numpy as np
import pandas as pd
import pytest

from utils.evaluation.TextAnalyzer import (
    TextAnalyzer,
    lexical_jaccard,
    normalized_token_edit_distance,
    token_edit_distance,
    word_ratio,
    wordration,
)


class FakeSemanticModel:
    """Small bag-of-words encoder used to keep tests offline and deterministic."""

    vocabulary = {"same": 0, "text": 1, "cat": 2, "dog": 3, "extra": 4}

    def encode(self, texts, **kwargs):
        embeddings = np.zeros((len(texts), len(self.vocabulary)), dtype=float)
        for row, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                if token in self.vocabulary:
                    embeddings[row, self.vocabulary[token]] += 1.0
        return embeddings


class FakeScoreAnalyzer:
    def __init__(self, frame):
        self.frame = frame

    def to_dataframe(self):
        return self.frame.copy()


def score_row(sample_id, regime, text, detector="detector-a"):
    return {
        "sample_id": sample_id,
        "dataset": "xsum",
        "length_band": "short",
        "generator": "llama",
        "rewriter": "llama" if regime in {"h2l", "llm2l"} else None,
        "regime": regime,
        "text": text,
        "detector": detector,
        "score": 0.5,
        "threshold": 0.7,
        "label": int(regime != "human"),
        "predicted_label": 0,
    }


def test_word_ratio_uses_rewritten_over_source_word_count():
    assert word_ratio("one two", "one two three") == pytest.approx(1.5)
    assert wordration("one two", "one two three") == pytest.approx(1.5)
    assert TextAnalyzer.word_ratio("one two", "one two three") == pytest.approx(1.5)
    assert np.isnan(word_ratio("", "one"))


def test_token_edit_distance_and_normalization():
    source = "the quick brown fox"
    rewritten = "the slow brown fox jumps"

    assert token_edit_distance(source, rewritten) == 2
    assert normalized_token_edit_distance(source, rewritten) == pytest.approx(2 / 5)
    assert normalized_token_edit_distance("", "") == 0.0
    assert normalized_token_edit_distance("", "hello") == 1.0


def test_lexical_jaccard_normalizes_case_and_ignores_punctuation():
    assert lexical_jaccard("The cat, sat", "the CAT slept") == pytest.approx(2 / 4)
    assert lexical_jaccard("", "") == 1.0
    assert TextAnalyzer.lexical_jaccard("Same!", "same") == 1.0


def test_semantic_similarity_uses_cosine_similarity():
    analyzer = TextAnalyzer(
        FakeScoreAnalyzer(pd.DataFrame()),
        semantic_model=FakeSemanticModel(),
    )

    assert analyzer.semantic_similarity("same text", "same text") == pytest.approx(1.0)
    assert analyzer.semantic_similarity("cat", "dog") == pytest.approx(0.0)


def test_pairwise_analysis_correlates_regimes_and_deduplicates_detectors():
    rows = [
        score_row("001", "human", "same text"),
        score_row("001", "human", "same text", detector="detector-b"),
        score_row("001", "h2l", "same text extra"),
        score_row("001", "h2l", "same text extra", detector="detector-b"),
        score_row("001", "free_llm", "cat"),
        score_row("001", "llm2l", "dog"),
        score_row("002", "human", "cat"),
        score_row("002", "h2l", "cat"),
    ]
    analyzer = TextAnalyzer(
        FakeScoreAnalyzer(pd.DataFrame(rows)),
        semantic_model=FakeSemanticModel(),
    )

    result = analyzer.to_dataframe()

    assert result.columns.tolist() == TextAnalyzer.COLUMNS
    assert len(result) == 3

    human_h2l = result[
        (result["sample_id"] == "001")
        & (result["source_regime"] == "human")
    ].iloc[0]
    assert human_h2l["source_text"] == "same text"
    assert human_h2l["rewritten_text"] == "same text extra"
    assert human_h2l["source_word_count"] == 2
    assert human_h2l["rewritten_word_count"] == 3
    assert human_h2l["word_ratio"] == pytest.approx(1.5)
    assert human_h2l["token_edit_distance"] == 1
    assert human_h2l["normalized_token_edit_distance"] == pytest.approx(1 / 3)
    assert human_h2l["lexical_jaccard"] == pytest.approx(2 / 3)
    assert 0.0 < human_h2l["semantic_similarity"] < 1.0

    free_llm_llm2l = result[
        (result["sample_id"] == "001")
        & (result["source_regime"] == "free_llm")
    ].iloc[0]
    assert free_llm_llm2l["rewritten_regime"] == "llm2l"
    assert free_llm_llm2l["token_edit_distance"] == 1
    assert free_llm_llm2l["normalized_token_edit_distance"] == 1.0
    assert free_llm_llm2l["semantic_similarity"] == pytest.approx(0.0)


def test_custom_rewrite_pair():
    frame = pd.DataFrame(
        [
            score_row("001", "human", "same text"),
            score_row("001", "free_llm", "same text extra"),
        ]
    )
    analyzer = TextAnalyzer(
        FakeScoreAnalyzer(frame),
        semantic_model=FakeSemanticModel(),
    )

    result = analyzer.to_dataframe(rewrite_pairs=[("human", "free_llm")])

    assert len(result) == 1
    assert result.loc[0, "source_regime"] == "human"
    assert result.loc[0, "rewritten_regime"] == "free_llm"
