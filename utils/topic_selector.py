from typing import Optional
import re
from blingfire import text_to_sentences


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def truncate_to_words(text: str, max_words: int) -> str:
    words = re.findall(r"\b[\w'-]+\b", text)
    return " ".join(words[:max_words])


def split_sentences(text: str) -> list[str]:
    text = normalize_whitespace(text)
    sent_text = text_to_sentences(text)
    return [
        s.strip()
        for s in sent_text.split("\n")
        if s.strip()
    ]


def extract_first_sentences_topic(
    text: str,
    n_sentences: int = 2,
    max_words: int = 40,
    min_words: int = 8,
) -> Optional[str]:
    sentences = split_sentences(text)

    if not sentences:
        return None

    topic = " ".join(sentences[:n_sentences])

    if word_count(topic) > max_words:
        topic = sentences[0]

    if word_count(topic) > max_words:
        topic = truncate_to_words(topic, max_words)

    topic = normalize_whitespace(topic)

    if word_count(topic) < min_words:
        return None

    return topic



def add_topic_to_dataset(dataset, topic_source: str):
    def _get_topic(batch, topic_source):
        """Add topics to a batched Hugging Face Dataset."""

        if topic_source == "first_sentences":
            topics = [
                extract_first_sentences_topic(text)
                for text in batch["text"]
            ]
        else:
            topics = batch[topic_source]

        return {"topic": topics}

    return dataset.map(
        _get_topic,
        fn_kwargs={"topic_source": topic_source},
        batched=True,
        batch_size=100
    )