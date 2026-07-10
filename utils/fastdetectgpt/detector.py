import os
from typing import Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


torch.set_grad_enabled(False)


def _normalize_option(value: str) -> str:
    return str(value).strip().lower()


def _resolve_dtype(dtype: str) -> torch.dtype:
    dtype = _normalize_option(dtype)
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp16", "float16"}:
        return torch.float16
    if dtype in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype for FastDetectGPT: {dtype}")


def get_bnb_config(quantization: str, compute_dtype: torch.dtype):
    quantization = _normalize_option(quantization)
    if quantization in {"none", "no", "false", "0", ""}:
        return None

    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "FastDetectGPT quantization requires bitsandbytes in this Python environment. "
            "Install it or set detector.quantization=none."
        ) from exc

    if quantization == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quantization == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    raise ValueError(
        "Unsupported FastDetectGPT quantization. Expected one of: none, int8, int4."
    )


def _model_input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


class FastDetectGPT:
    def __init__(
        self,
        sampling_model_name_or_path: str,
        scoring_model_name_or_path: str,
        max_token_observed: int = 512,
        dtype: str = "bf16",
        quantization: str = "none",
        trust_remote_code: bool = False,
        device_map: str = "auto",
        local_files_only: bool = False,
    ):
        self.max_token_observed = int(max_token_observed)
        self.scoring_model_name_or_path = scoring_model_name_or_path
        self.sampling_model_name_or_path = sampling_model_name_or_path
        self.same_model = sampling_model_name_or_path == scoring_model_name_or_path

        torch_dtype = _resolve_dtype(dtype)
        quantization_config = get_bnb_config(quantization, torch_dtype)

        token = os.environ.get("HF_TOKEN", None)
        tokenizer_kwargs = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        if token:
            tokenizer_kwargs["token"] = token

        self.scoring_tokenizer = AutoTokenizer.from_pretrained(
            scoring_model_name_or_path,
            **tokenizer_kwargs,
        )
        if self.scoring_tokenizer.pad_token_id is None:
            self.scoring_tokenizer.pad_token = self.scoring_tokenizer.eos_token
        self.scoring_tokenizer.padding_side = "right"

        if self.same_model:
            self.sampling_tokenizer = self.scoring_tokenizer
        else:
            self.sampling_tokenizer = AutoTokenizer.from_pretrained(
                sampling_model_name_or_path,
                **tokenizer_kwargs,
            )
            if self.sampling_tokenizer.pad_token_id is None:
                self.sampling_tokenizer.pad_token = self.sampling_tokenizer.eos_token
            self.sampling_tokenizer.padding_side = "right"

        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        if token:
            model_kwargs["token"] = token
        if device_map and _normalize_option(device_map) != "none":
            model_kwargs["device_map"] = device_map
        if quantization_config is None:
            model_kwargs["dtype"] = torch_dtype
        else:
            model_kwargs["quantization_config"] = quantization_config

        self.scoring_model = AutoModelForCausalLM.from_pretrained(
            scoring_model_name_or_path,
            **model_kwargs,
        )
        self.scoring_model.eval()

        if self.same_model:
            self.sampling_model = self.scoring_model
        else:
            self.sampling_model = AutoModelForCausalLM.from_pretrained(
                sampling_model_name_or_path,
                **model_kwargs,
            )
            self.sampling_model.eval()

    def _sanitize_texts(self, texts: list[str]) -> list[str]:
        fallback = self.scoring_tokenizer.eos_token or " "
        return [text if str(text).strip() else fallback for text in texts]

    def _tokenize(self, tokenizer, texts: list[str]):
        return tokenizer(
            self._sanitize_texts(texts),
            return_tensors="pt",
            padding="longest" if len(texts) > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_attention_mask=True,
            return_token_type_ids=False,
        )

    def _tokenize_for_models(self, texts: list[str]):
        scoring_encodings = self._tokenize(self.scoring_tokenizer, texts)
        if self.same_model:
            return scoring_encodings, scoring_encodings

        sampling_encodings = self._tokenize(self.sampling_tokenizer, texts)
        if scoring_encodings["input_ids"].shape != sampling_encodings["input_ids"].shape:
            raise ValueError(
                "FastDetectGPT requires sampling and scoring tokenizers to produce "
                "the same token IDs for the same text."
            )
        if not torch.equal(scoring_encodings["input_ids"], sampling_encodings["input_ids"]):
            raise ValueError(
                "FastDetectGPT tokenizer mismatch: sampling and scoring models "
                "produced different token IDs."
            )
        return scoring_encodings, sampling_encodings

    @staticmethod
    def _align_vocab(logits_ref, logits_score, labels):
        if logits_ref.size(-1) == logits_score.size(-1):
            return logits_ref, logits_score

        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        if labels.numel() and int(labels.max().item()) >= vocab_size:
            raise ValueError(
                "FastDetectGPT cannot align model vocabularies because an observed "
                "token ID is outside the shared vocabulary range."
            )
        return logits_ref[:, :, :vocab_size], logits_score[:, :, :vocab_size]

    @staticmethod
    def _sampling_discrepancy_analytic(logits_ref, logits_score, labels, valid_mask):
        logits_ref, logits_score = FastDetectGPT._align_vocab(
            logits_ref,
            logits_score,
            labels,
        )

        labels = labels.to(logits_score.device).unsqueeze(-1)
        valid_mask = valid_mask.to(logits_score.device)
        mask = valid_mask.float()

        lprobs_score = torch.log_softmax(logits_score.float(), dim=-1)
        probs_ref = torch.softmax(logits_ref.float(), dim=-1)

        log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
        mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
        var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
        var_ref = torch.clamp(var_ref, min=0.0)

        seq_log_likelihood = (log_likelihood * mask).sum(dim=-1)
        seq_mean_ref = (mean_ref * mask).sum(dim=-1)
        seq_var_ref = (var_ref * mask).sum(dim=-1)

        scores = (seq_log_likelihood - seq_mean_ref) / torch.sqrt(
            torch.clamp(seq_var_ref, min=1e-8)
        )
        has_tokens = valid_mask.sum(dim=-1) > 0
        return torch.where(has_tokens, scores, torch.zeros_like(scores))

    @torch.inference_mode()
    def compute_score(self, input_text: Union[str, list[str]]) -> Union[float, list[float]]:
        is_single = isinstance(input_text, str)
        texts = [input_text] if is_single else list(input_text)
        if not texts:
            return [] if not is_single else 0.0

        scoring_encodings, sampling_encodings = self._tokenize_for_models(texts)

        scoring_device = _model_input_device(self.scoring_model)
        scoring_encodings = scoring_encodings.to(scoring_device)
        input_ids = scoring_encodings["input_ids"]
        attention_mask = scoring_encodings["attention_mask"]
        labels = input_ids[:, 1:]
        valid_mask = attention_mask[:, 1:].bool()

        logits_score = self.scoring_model(**scoring_encodings).logits[:, :-1, :]

        if self.same_model:
            logits_ref = logits_score
        else:
            sampling_device = _model_input_device(self.sampling_model)
            sampling_encodings = sampling_encodings.to(sampling_device)
            logits_ref = self.sampling_model(**sampling_encodings).logits[:, :-1, :]
            logits_ref = logits_ref.to(logits_score.device)

        scores = self._sampling_discrepancy_analytic(
            logits_ref=logits_ref,
            logits_score=logits_score,
            labels=labels,
            valid_mask=valid_mask,
        )
        scores = scores.detach().float().cpu().tolist()
        return scores[0] if is_single else scores
