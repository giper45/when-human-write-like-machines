from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from typing import Any, Optional

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

from utils.evaluation.ScoreAnalyzer import ScoreAnalyzer


DEFAULT_SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_REWRITE_PAIRS = (("human", "h2l"), ("free_llm", "llm2l"))

_WORD_PATTERN = re.compile(r"\b\w+(?:['’-]\w+)*\b", flags=re.UNICODE)
_TOKEN_PATTERN = re.compile(r"\w+(?:['’-]\w+)*|[^\w\s]", flags=re.UNICODE)


def word_tokens(text: str) -> list[str]:
    """Tokenize text into words, excluding standalone punctuation."""
    return _WORD_PATTERN.findall(text)


def edit_tokens(text: str) -> list[str]:
    """Tokenize text into words and punctuation for token-level edit distance."""
    return _TOKEN_PATTERN.findall(text)


def lexical_tokens(text: str) -> list[str]:
    """Return case-folded, Unicode-normalized lexical tokens."""
    return [
        unicodedata.normalize("NFKC", token).casefold()
        for token in word_tokens(text)
    ]


def word_ratio(source_text: str, rewritten_text: str) -> float:
    """Return rewritten/source word-count ratio."""
    source_count = len(word_tokens(source_text))
    if source_count == 0:
        return np.nan
    return len(word_tokens(rewritten_text)) / source_count


# Alias matching the name used in the initial analysis specification.
wordration = word_ratio


def token_edit_distance(source_text: str, rewritten_text: str) -> int:
    """Return Levenshtein distance between source and rewritten token sequences."""
    return int(Levenshtein.distance(edit_tokens(source_text), edit_tokens(rewritten_text)))


def normalized_token_edit_distance(source_text: str, rewritten_text: str) -> float:
    """Return token edit distance divided by the longest token sequence."""
    source_tokens = edit_tokens(source_text)
    rewritten_tokens = edit_tokens(rewritten_text)
    denominator = max(len(source_tokens), len(rewritten_tokens))
    if denominator == 0:
        return 0.0
    return float(Levenshtein.distance(source_tokens, rewritten_tokens) / denominator)


def lexical_jaccard(source_text: str, rewritten_text: str) -> float:
    """Return Jaccard overlap between normalized lexical-token sets."""
    source_vocabulary = set(lexical_tokens(source_text))
    rewritten_vocabulary = set(lexical_tokens(rewritten_text))
    union = source_vocabulary | rewritten_vocabulary
    if not union:
        return 1.0
    return len(source_vocabulary & rewritten_vocabulary) / len(union)


class TextAnalyzer:
    """Analyze correlated source/rewrite pairs produced by ``ScoreAnalyzer``."""

    word_ratio = staticmethod(word_ratio)
    wordration = staticmethod(word_ratio)
    token_edit_distance = staticmethod(token_edit_distance)
    normalized_token_edit_distance = staticmethod(normalized_token_edit_distance)
    lexical_jaccard = staticmethod(lexical_jaccard)

    COLUMNS = [
        "sample_id",
        "dataset",
        "length_band",
        "generator",
        "rewriter",
        "source_regime",
        "rewritten_regime",
        "source_text",
        "rewritten_text",
        "source_word_count",
        "rewritten_word_count",
        "word_ratio",
        "token_edit_distance",
        "normalized_token_edit_distance",
        "lexical_jaccard",
        "semantic_similarity",
    ]

    def __init__(
        self,
        score_analyzer: ScoreAnalyzer,
        *,
        semantic_model_name: str = DEFAULT_SEMANTIC_MODEL,
        semantic_model: Optional[Any] = None,
        batch_size: int = 32,
        device: Optional[str] = None,
        show_progress_bar: bool = False,
    ) -> None:
        self.score_analyzer = score_analyzer
        self.semantic_model_name = semantic_model_name
        self._semantic_model = semantic_model
        self.batch_size = batch_size
        self.device = device
        self.show_progress_bar = show_progress_bar

    @property
    def semantic_model(self):
        """Load the SentenceTransformer only when semantic scores are requested."""
        if self._semantic_model is None:
            from sentence_transformers import SentenceTransformer

            self._semantic_model = SentenceTransformer(
                self.semantic_model_name,
                device=self.device,
            )
        return self._semantic_model

    def semantic_similarity(self, source_text: str, rewritten_text: str) -> float:
        """Return cosine similarity between sentence embeddings of two texts."""
        return float(self._semantic_similarities([source_text], [rewritten_text])[0])

    def to_dataframe(
        self,
        rewrite_pairs: Iterable[tuple[str, str]] = DEFAULT_REWRITE_PAIRS,
    ) -> pd.DataFrame:
        """Return text metrics for each correlated source/rewrite pair."""
        score_frame = self.score_analyzer.to_dataframe()
        paired_frame = self._build_pairs(score_frame, rewrite_pairs)
        if paired_frame.empty:
            return pd.DataFrame(columns=self.COLUMNS)

        source_texts = paired_frame["source_text"].tolist()
        rewritten_texts = paired_frame["rewritten_text"].tolist()
        source_tokens = [edit_tokens(text) for text in source_texts]
        rewritten_tokens = [edit_tokens(text) for text in rewritten_texts]

        paired_frame["source_word_count"] = [
            len(word_tokens(text)) for text in source_texts
        ]
        paired_frame["rewritten_word_count"] = [
            len(word_tokens(text)) for text in rewritten_texts
        ]
        paired_frame["word_ratio"] = [
            word_ratio(source, rewritten)
            for source, rewritten in zip(source_texts, rewritten_texts)
        ]

        distances = [
            int(Levenshtein.distance(source, rewritten))
            for source, rewritten in zip(source_tokens, rewritten_tokens)
        ]
        paired_frame["token_edit_distance"] = distances
        paired_frame["normalized_token_edit_distance"] = [
            distance / max(len(source), len(rewritten))
            if source or rewritten
            else 0.0
            for distance, source, rewritten in zip(
                distances, source_tokens, rewritten_tokens
            )
        ]
        paired_frame["lexical_jaccard"] = [
            lexical_jaccard(source, rewritten)
            for source, rewritten in zip(source_texts, rewritten_texts)
        ]
        paired_frame["semantic_similarity"] = self._semantic_similarities(
            source_texts,
            rewritten_texts,
        )
        # Hugging Face datasets may expose IDs as integers in one dataset and
        # strings in another. A uniform nullable string dtype keeps the combined
        # frame serializable by Arrow/Parquet without changing ID semantics.
        paired_frame["sample_id"] = paired_frame["sample_id"].astype("string")

        return paired_frame.loc[:, self.COLUMNS].reset_index(drop=True)

    def analyze(
        self,
        rewrite_pairs: Iterable[tuple[str, str]] = DEFAULT_REWRITE_PAIRS,
    ) -> pd.DataFrame:
        return self.to_dataframe(rewrite_pairs=rewrite_pairs)

    def _build_pairs(
        self,
        score_frame: pd.DataFrame,
        rewrite_pairs: Iterable[tuple[str, str]],
    ) -> pd.DataFrame:
        required_columns = {
            "sample_id",
            "dataset",
            "length_band",
            "generator",
            "rewriter",
            "regime",
            "text",
        }
        missing_columns = required_columns - set(score_frame.columns)
        if missing_columns:
            raise ValueError(
                f"ScoreAnalyzer frame is missing columns: {sorted(missing_columns)}"
            )

        identity_columns = ["sample_id", "dataset", "generator", "regime"]
        text_rows = score_frame[
            identity_columns + ["length_band", "rewriter", "text"]
        ].drop_duplicates()
        conflicting = (
            text_rows.groupby(identity_columns, dropna=False)["text"]
            .nunique(dropna=False)
            .gt(1)
        )
        if conflicting.any():
            identities = conflicting[conflicting].index.tolist()
            raise ValueError(
                "Multiple texts found for the same sample/dataset/generator/regime: "
                f"{identities[:5]}"
            )
        text_rows = text_rows.drop_duplicates(subset=identity_columns, keep="first")

        join_columns = ["sample_id", "dataset", "generator"]
        pair_frames = []
        for source_regime, rewritten_regime in rewrite_pairs:
            source = text_rows[text_rows["regime"] == source_regime][
                join_columns + ["length_band", "text"]
            ].rename(columns={"text": "source_text"})
            rewritten = text_rows[text_rows["regime"] == rewritten_regime][
                join_columns + ["rewriter", "text"]
            ].rename(columns={"text": "rewritten_text"})

            pair = source.merge(
                rewritten,
                on=join_columns,
                how="inner",
                validate="one_to_one",
            )
            pair["source_regime"] = source_regime
            pair["rewritten_regime"] = rewritten_regime
            pair_frames.append(pair)

        if not pair_frames:
            return pd.DataFrame(columns=self.COLUMNS)
        return pd.concat(pair_frames, ignore_index=True)

    def _semantic_similarities(
        self,
        source_texts: Sequence[str],
        rewritten_texts: Sequence[str],
    ) -> np.ndarray:
        if len(source_texts) != len(rewritten_texts):
            raise ValueError("Source and rewritten text collections must have equal length.")
        if not source_texts:
            return np.asarray([], dtype=float)

        text_count = len(source_texts)
        embeddings = self.semantic_model.encode(
            list(source_texts) + list(rewritten_texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=self.show_progress_bar,
        )
        embeddings = np.asarray(embeddings, dtype=float)
        source_embeddings = embeddings[:text_count]
        rewritten_embeddings = embeddings[text_count:]

        numerator = np.sum(source_embeddings * rewritten_embeddings, axis=1)
        denominator = np.linalg.norm(source_embeddings, axis=1) * np.linalg.norm(
            rewritten_embeddings, axis=1
        )
        similarities = np.divide(
            numerator,
            denominator,
            out=np.full(text_count, np.nan, dtype=float),
            where=denominator != 0,
        )
        return np.clip(similarities, -1.0, 1.0)
