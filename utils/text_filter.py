from enum import Enum
from pathlib import Path
from typing import Optional, TypedDict
from tqdm import tqdm
from omegaconf import DictConfig
from pydantic import BaseModel, ValidationError
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline, AutoModelForCausalLM
from transformers import pipeline as hf_pipeline
from utils.models import FilteringReport
from utils.device import get_quantization_config
from utils.logger import log
from datasets import Dataset
import outlines
import re


CONTROL_CHARS = re.compile(r"[\x00-\x1F\x7F]")
NUM_RETRIES = 20  # Numero di tentativi di retry per la classificazione JSON

def clean_json_string(s: str) -> str:
    """Rimuove caratteri di controllo non validi in JSON."""
    return CONTROL_CHARS.sub("", s)


def remove_preamble(s: str) -> str:
    """Remove until Text: string"""
    match = re.search(r"Text:\s*", s)
    if match:
        return s[match.end():].strip()
    return s.strip()

def remove_newlines(s: str) -> str:
    """Remove newlines and excessive spaces."""
    return re.sub(r"\s+", " ", s).strip()

def get_cleaned_text(s: str) -> str:
    s = remove_preamble(s)
    s = remove_newlines(s)
    return s

# ---- Schema definition ----
class Classification(BaseModel):
    classification: str
    explanation: str

    def __str__(self):
        return f"Classification: {self.classification}\nExplanation: {self.explanation}"    

class ClassificationText(TypedDict):
    index: int
    text: str
    classification: str
    explanation: str


class TokenLengthTruncator:
    """
    Class to filter texts based on their token length using a specified tokenizer.
    """

    def __init__(self, model_name="bert-base-uncased", max_length=300):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def filter(self, batch):
        tokenized = self.tokenizer(batch["text"], truncation=False)
        lengths = [len(ids) for ids in tokenized["input_ids"]]
        return [150 <= l <= 300 for l in lengths]

class TextWordLengthTruncator:
    """
    Class to filter texts based on their word count.
    """

    def __init__(self, min_length=150, max_length=300):
        self.min_length = min_length
        self.max_length = max_length

    def add_lengths(self, batch):
        return {
            "word_length": [len(text.split()) for text in batch["text"]]
        }


    def filter(self, batch):
        lengths = [len(text.split()) for text in batch["text"]]
        return [self.min_length <= l <= self.max_length for l in lengths]


class DatasetColumnFilter:
    """
    Class to filter dataset columns based on a specified column name.
    """

    def __init__(self, column_name="text"):
        self.column_name = column_name
        self.columns_to_rename = ["document", "prompt"]
        self.columns_to_keep = [self.column_name]

    def filter(self, dataset):
        for col in self.columns_to_rename:
            if col in dataset.column_names:
                dataset = dataset.rename_column(col, self.column_name)
        return dataset.select_columns(self.columns_to_keep)

class ToxicityFilter:
    """
    Filter texts using Hugging Face classifier (fast, batched).
    """

    def __init__(self, 
                 model_path="martin-ha/toxic-comment-model", 
                 threshold=0.5,
                 max_length: int = 512):

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, device_map="auto")
        self.pipeline = TextClassificationPipeline(
            model=model,
            tokenizer=tokenizer,
            truncation=True,
            max_length=max_length,
        )
        self.threshold = threshold

    def filter(self, batch):
        preds = self.pipeline(batch["text"])
        return [float(p["score"]) > self.threshold for p in preds]


class NotAcceptableFilter:
    """
    Filter texts using quantized LLaMA-3 and Outlines for structured JSON output.
    """

    def __init__(self, model_id="meta-llama/Llama-3.2-3B-Instruct",
                 max_new_tokens=512):

        log.info("Loading quantized model with Outlines...")

        # quantization per risparmiare VRAM/RAM
        quant_config = get_quantization_config()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})


        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto"
        )
        # self.model.resize_token_embeddings(len(self.tokenizer))

        self.max_new_tokens = max_new_tokens
        self.generator = outlines.models.Transformers(
            model=self.model,
            tokenizer=self.tokenizer,
        )

        # wrap con outlines
        self.generator = outlines.from_transformers(self.model, self.tokenizer)


    def get_prompt(self, text: str) -> str:
        """
        Costruisce il prompt di classificazione per un singolo testo.
        """
        return (
            "Classify the following text as 'Acceptable' or 'NotAcceptable'.\n"
            "A text is 'NotAcceptable' if it promotes hate, violence, discrimination, "
            "harmful stereotypes, dangerous misinformation, harassment, or contains "
            "content that should not be paraphrased "
            "(e.g., historical quotes promoting ideologies, sensitive legal texts, private/confidential info).\n\n"
            "Return a valid JSON object with this exact format:\n\n"
            "{\n"
            '  "classification": "Acceptable" | "NotAcceptable",\n'
            '  "explanation": "your explanation here"\n'
            "}\n\n"
            "Do not add anything else, and remove characters that cannot be parsed in JSON.\n\n"
            f"Text: {text}\nAnswer:"
        )
    def classify(self, text: str) -> Classification:
        """
        Restituisce un JSON del tipo:
        {
            "classification": "Acceptable" | "NotAcceptable",
            "explanation": "..."
        }
        """
        prompt  = self.get_prompt(text)

        result = self.generator(prompt, Classification, max_new_tokens=self.max_new_tokens)
        result = Classification.model_validate_json(result)
        return result


    def get_classified_texts(self, batch, indices) -> dict[str, list]:
        """
        Funziona con dataset.map(..., batched=True, with_indices=True).
        Restituisce un dict con liste parallele (stessa lunghezza di batch["text"]).
        """
        texts = batch["text"]
        prompts = [self.get_prompt(t) for t in texts]
        results = []
        retry_texts = []
        retry_indices = []

        # primo tentativo (batch)
        outs = self.generator.batch(prompts, Classification, max_new_tokens=self.max_new_tokens)

        for i, out in enumerate(outs):
            try:
                cleaned = clean_json_string(out)
                obj = Classification.model_validate_json(cleaned)
                results.append({
                    "index": indices[i],
                    "text": texts[i],
                    "classification": obj.classification,
                    "explanation": obj.explanation
                })
            except ValidationError:
                # fallback retry
                retry_texts.append(texts[i])
                retry_indices.append(i)
                results.append(None)

        # retry multipli
        for attempt in range(NUM_RETRIES):
            if not retry_texts:
                break
            retry_outs = self.generator.batch(retry_texts, Classification, max_new_tokens=self.max_new_tokens)
            new_retry_texts = []
            new_retry_indices = []
            for j, out in enumerate(retry_outs):
                try:
                    cleaned = clean_json_string(out)
                    obj = Classification.model_validate_json(cleaned)
                    results[retry_indices[j]] = {
                        "index": indices[retry_indices[j]],
                        "text": retry_texts[j],
                        "classification": obj.classification,
                        "explanation": obj.explanation
                    }
                except ValidationError as e:
                    log.error(f"Retry {attempt} failed for index {retry_indices[j]}:", e)
                    new_retry_texts.append(retry_texts[j])
                    new_retry_indices.append(retry_indices[j])
            retry_texts, retry_indices = new_retry_texts, new_retry_indices

        # fallback finale: riempi i None con valori di default
        for i, res in enumerate(results):
            if res is None:
                results[i] = {
                    "index": indices[i],
                    "text": texts[i],
                    "classification": "Unknown",
                    "explanation": "Failed to classify after retries"
                }

        # HuggingFace .map richiede dict[str, list]
        return {
            "index": [r["index"] for r in results],
            "text": [get_cleaned_text(r["text"]) for r in results],
            "classification": [r["classification"] for r in results],
            "explanation": [r["explanation"] for r in results],
        }
    

    def filter(self, batch):
        texts = batch["text"]
        prompts = [self.get_prompt(t) for t in texts]

        # primo tentativo in batch
        outs = self.generator.batch(prompts, Classification, max_new_tokens=self.max_new_tokens)

        results = []
        retry_prompts = []
        retry_indices = []

        for i, out in enumerate(outs):
            try:
                cleaned = clean_json_string(out)
                obj = Classification.model_validate_json(cleaned)
                results.append(obj)
            except ValidationError:
                # segno quelli falliti
                retry_prompts.append(prompts[i])
                retry_indices.append(i)
                results.append(None)

        # retry fino a 2 volte sugli item falliti
        for attempt in range(NUM_RETRIES):
            if not retry_prompts:
                break
            retry_outs = self.generator.batch(retry_prompts, Classification, max_new_tokens=self.max_new_tokens)
            new_retry_prompts = []
            new_retry_indices = []
            for j, out in enumerate(retry_outs):
                try:
                    log.info("out: ", out)
                    obj = Classification.model_validate_json(out)
                    results[retry_indices[j]] = obj
                except ValidationError as e:
                    log.error(f"Retry {attempt} failed for index {retry_indices[j]}:", e)
                    new_retry_prompts.append(retry_prompts[j])
                    new_retry_indices.append(retry_indices[j])
            retry_prompts, retry_indices = new_retry_prompts, new_retry_indices

        # fallback finale: se ancora None → scarta
        bool_results = []
        for obj in results:
            if obj is None:
                bool_results.append(False)
            elif obj.classification.lower().replace(" ", "") == "notacceptable":
                bool_results.append(False)
            else:
                bool_results.append(True)

        return bool_results





import re
from dataclasses import dataclass


@dataclass
class ArtifactFilterConfig:
    # HTML / markup
    max_html_tag_count: int = 5
    max_html_char_ratio: float = 0.05

    # newline / formatting
    max_newline_count: int = 12
    max_newline_ratio: float = 0.08

    # lists / tables / code
    max_bullet_lines: int = 3
    max_table_like_lines: int = 2
    max_code_like_lines: int = 2

    # truncation
    reject_truncated: bool = True

    # prompt/source artifacts
    reject_prompt_leakage: bool = True


class StructuralArtifactFilter:
    """
    Deterministic filter for excluding texts with heavy markup, lists, tables,
    code, excessive newlines, truncation artifacts, or prompt/source leakage.

    Compatible with HuggingFace dataset.filter(..., batched=True).
    """

    HTML_TAG_RE = re.compile(
        r"</?\s*[a-zA-Z][a-zA-Z0-9]*(?:\s+[^>]*)?>",
        re.IGNORECASE,
    )

    HTML_ENTITY_RE = re.compile(
        r"&(?:amp|lt|gt|quot|apos|nbsp|#\d+|#x[0-9a-fA-F]+);"
    )

    BULLET_LINE_RE = re.compile(
        r"^\s*(?:[-*•‣▪]|\d+[.)]|[a-zA-Z][.)])\s+"
    )

    TABLE_LINE_RE = re.compile(
        r"^\s*\|.*\|\s*$|(?:\t.*\t)|(?:\S+\s{2,}\S+\s{2,}\S+)"
    )

    CODE_LINE_RE = re.compile(
        r"^\s*(?:"
        r"```|"
        r"def\s+\w+\s*\(|"
        r"class\s+\w+\s*[:\(]|"
        r"import\s+\w+|"
        r"from\s+\w+\s+import\s+|"
        r"for\s+.+\s+in\s+.+:|"
        r"while\s+.+:|"
        r"if\s+.+:|"
        r"return\s+.+|"
        r"\{|\}|\[|\]|"
        r"#include\s+|"
        r"console\.log\(|"
        r"function\s+\w+\s*\("
        r")"
    )

    TRUNCATION_RE = re.compile(
        r"(?:"
        r"\.\.\.\s*$|"
        r"\[…\]\s*$|"
        r"\[truncated\]|"
        r"\(truncated\)|"
        r"read more\s*$|"
        r"continue reading\s*$|"
        r"continued\s*$|"
        r"\bmore\.\.\.\s*$"
        r")",
        re.IGNORECASE,
    )

    PROMPT_LEAKAGE_RE = re.compile(
        r"(?:"
        r"^text:\s*|"
        r"^topic:\s*|"
        r"^answer:\s*|"
        r"^rewrite the following text|"
        r"^write a fluent|"
        r"^constraints:\s*|"
        r"return only the|"
        r"do not add explanations|"
        r"preserve the original meaning|"
        r"as an ai language model|"
        r"here is the rewritten text|"
        r"sure[,! ]|"
        r"certainly[,! ]|"
        r"i cannot assist"
        r")",
        re.IGNORECASE,
    )

    SOURCE_ARTIFACT_RE = re.compile(
        r"(?:"
        r"copyright\s+\d{4}|"
        r"all rights reserved|"
        r"subscribe to continue|"
        r"sign in to continue|"
        r"advertisement|"
        r"cookie policy|"
        r"privacy policy|"
        r"terms of service|"
        r"related articles|"
        r"click here|"
        r"image caption|"
        r"photo credit|"
        r"share this article"
        r")",
        re.IGNORECASE,
    )

    def __init__(self, config: ArtifactFilterConfig | None = None):
        self.config = config or ArtifactFilterConfig()

    def count_html_tags(self, text: str) -> int:
        return len(self.HTML_TAG_RE.findall(text))

    def html_char_ratio(self, text: str) -> float:
        if not text:
            return 0.0

        html_parts = self.HTML_TAG_RE.findall(text)
        entity_parts = self.HTML_ENTITY_RE.findall(text)
        html_chars = sum(len(x) for x in html_parts + entity_parts)

        return html_chars / max(len(text), 1)

    def newline_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        return text.count("\n") / max(len(text), 1)

    def line_counts(self, text: str) -> dict:
        lines = [line for line in text.splitlines() if line.strip()]

        bullet_lines = sum(
            bool(self.BULLET_LINE_RE.search(line))
            for line in lines
        )

        table_like_lines = sum(
            bool(self.TABLE_LINE_RE.search(line))
            for line in lines
        )

        code_like_lines = sum(
            bool(self.CODE_LINE_RE.search(line))
            for line in lines
        )

        return {
            "n_lines": len(lines),
            "bullet_lines": bullet_lines,
            "table_like_lines": table_like_lines,
            "code_like_lines": code_like_lines,
        }

    def reject_reason(self, text: str) -> str | None:
        if text is None or not str(text).strip():
            return "empty"

        text = str(text)
        cfg = self.config

        html_tag_count = self.count_html_tags(text)
        html_ratio = self.html_char_ratio(text)

        if html_tag_count > cfg.max_html_tag_count:
            return "heavy_html_tag_count"

        if html_ratio > cfg.max_html_char_ratio:
            return "heavy_html_ratio"

        newline_count = text.count("\n")
        newline_ratio = self.newline_ratio(text)

        if newline_count > cfg.max_newline_count:
            return "too_many_newlines"

        if newline_ratio > cfg.max_newline_ratio:
            return "high_newline_ratio"

        counts = self.line_counts(text)

        if counts["bullet_lines"] > cfg.max_bullet_lines:
            return "list_like_text"

        if counts["table_like_lines"] > cfg.max_table_like_lines:
            return "table_like_text"

        if counts["code_like_lines"] > cfg.max_code_like_lines:
            return "code_like_text"

        if cfg.reject_truncated and self.TRUNCATION_RE.search(text):
            return "truncated_text"

        if cfg.reject_prompt_leakage and self.PROMPT_LEAKAGE_RE.search(text):
            return "prompt_leakage"

        if self.SOURCE_ARTIFACT_RE.search(text):
            return "source_artifact"

        return None

    def is_valid(self, text: str) -> bool:
        return self.reject_reason(text) is None

    def filter(self, batch):
        return [self.is_valid(text) for text in batch["text"]]

    def annotate(self, batch):
        """
        Useful with dataset.map(..., batched=True) to keep diagnostics.
        """
        reasons = [self.reject_reason(text) for text in batch["text"]]
        return {
            "artifact_valid": [r is None for r in reasons],
            "artifact_reject_reason": [r or "" for r in reasons],
        }


def apply_text_filters(
    dataset: Dataset,
    filter: DictConfig,
    output_dir: Optional[str | Path] = None,
    dataset_name: str = "dataset",
) -> tuple[Dataset, FilteringReport]:
    """
    Apply the full filtering pipeline:
    1. word-length filter
    2. structural artifact filter
    3. Toxicity filter
    4. NotAcceptable LLM-based filter

    Returns the filtered dataset and a FilteringReport with per-step statistics.
    If ``output_dir`` is provided, the report is saved as
    ``<output_dir>/filtering_report.json``.
    """

    filtered_dataset = dataset
    report = FilteringReport(
        dataset_name=dataset_name,
        rows_initial=len(dataset),
        rows_final=len(dataset),
    )

    steps = [
        ("word_length",        filter.apply_word_length),
        ("structural_artifact", filter.apply_artifact_filter),
        ("toxicity",           filter.apply_toxicity_filter),
        ("not_acceptable",     filter.apply_not_acceptable_filter),
    ]
    active_steps = [(name, enabled) for name, enabled in steps if enabled]

    with tqdm(total=len(active_steps), desc="Filtering pipeline", unit="filter") as pbar:

        if filter.apply_word_length:
            pbar.set_description("word_length")
            log.info("Applying word-length filter...")
            rows_before = len(filtered_dataset)
            word_length_filter = TextWordLengthTruncator(
                min_length=filter.min_words,
                max_length=filter.max_words,
            )
            filtered_dataset = filtered_dataset.filter(
                word_length_filter.filter,
                batched=True,
                batch_size=filter.batch_size,
                load_from_cache_file=False,
            )
            report.add_step("word_length", rows_before, len(filtered_dataset))
            log.info(f"After word-length filter: {len(filtered_dataset)} examples")
            pbar.update(1)

        if filter.apply_artifact_filter:
            pbar.set_description("structural_artifact")
            log.info("Applying structural artifact filter...")
            rows_before = len(filtered_dataset)
            artifact_filter = StructuralArtifactFilter()
            filtered_dataset = filtered_dataset.filter(
                artifact_filter.filter,
                batched=True,
                batch_size=filter.batch_size,
                load_from_cache_file=False,
            )
            report.add_step("structural_artifact", rows_before, len(filtered_dataset))
            log.info(f"After artifact filter: {len(filtered_dataset)} examples")
            pbar.update(1)

        if filter.apply_toxicity_filter:
            pbar.set_description("toxicity")
            log.info("Applying toxicity filter...")
            rows_before = len(filtered_dataset)
            toxicity_filter = ToxicityFilter()
            filtered_dataset = filtered_dataset.filter(
                toxicity_filter.filter,
                batched=True,
                batch_size=filter.batch_size,
                load_from_cache_file=False,
            )
            report.add_step("toxicity", rows_before, len(filtered_dataset))
            log.info(f"After toxicity filter: {len(filtered_dataset)} examples")
            pbar.update(1)
        if filter.apply_not_acceptable_filter:
            pbar.set_description("not_acceptable")
            log.info("Applying NotAcceptable filter...")
            rows_before = len(filtered_dataset)
            not_acceptable_filter = NotAcceptableFilter()
            filtered_dataset = filtered_dataset.filter(
                not_acceptable_filter.filter,
                batched=True,
                batch_size=1,
                load_from_cache_file=False,
            )
            report.add_step("not_acceptable", rows_before, len(filtered_dataset))
            log.info(f"After NotAcceptable filter: {len(filtered_dataset)} examples")
            pbar.update(1)

        pbar.set_description("done")

    if output_dir is not None:
        dest = report.save(output_dir)
        log.info(f"Filtering report saved to {dest}")

    return filtered_dataset, report