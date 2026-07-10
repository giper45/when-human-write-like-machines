import os
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import DataCollatorWithPadding
from transformers import RobertaForSequenceClassification, RobertaTokenizer

from torch.utils.data import DataLoader
import transformers
import numpy as np
from utils.device import get_device
from utils.evaluation import Predictions

# from m4gt_classifier_train_optimized import softmax_with_temperature




def ensure_determinism(seed=42):
    """Ensure deterministic behavior across runs."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def get_data_loader(dataset, tokenizer, tokenized_fn, batch_size=32):
    """
    Creates a DataLoader with a Data Collator to ensure correct batching.
    """
    test_dataset = tokenized_datasets  # Assume train dataset is used
    tokenized_datasets = tokenized_fn(dataset, tokenizer)

    # Create a data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer)

    # Create DataLoader with collation
    data_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=data_collator)

    return data_loader




def predict_batch(data_loader, model, device, inverse_labels=False):
    """
    Runs batch inference using a DataLoader.
    """
    predicted_labels = []
    true_labels = []
    scores = []
    raw_scores = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Processing batches"):
            batch = {key: val.to(device) for key, val in batch.items()}  # Move batch to device
            labels = batch['labels'].cpu().tolist() 
            inputs = {key: batch[key] for key in batch if key != 'labels'}
            try:
                outputs = model(**inputs)
                logits = outputs.logits
                class_1_probs = torch.nn.functional.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                ai_scores = 1.0 - class_1_probs if inverse_labels else class_1_probs
                preds = (ai_scores >= 0.5).astype(int).tolist()

            except RuntimeError as e:
                print(f"CUDA ERROR: {e}")
                print(f"Batch keys: {batch.keys()}")
                print(f"Input shapes: {[batch[key].shape for key in batch]}")
                raise e  # Force stop to debug


            predicted_labels.extend(preds)
            true_labels.extend(labels)
            raw_scores.extend(ai_scores.tolist())
            scores.extend(ai_scores.tolist())

    return Predictions(
        predicted_labels=predicted_labels,
        true_labels=true_labels,
        pred_probs=scores,
        scores=scores,
        raw_scores=raw_scores,
        default_threshold=0.5,
        score_direction="higher_is_ai",
        scores_are_probabilities=True,
        metadata={"inverse_labels": bool(inverse_labels)},
    )


def get_batch_size(cfg):
    return int(cfg.experiment.batch_size)


def get_binoculars_batch_size(cfg):
    return int(getattr(cfg.detector, "batch_size", get_batch_size(cfg)))


def get_fastdetectgpt_batch_size(cfg):
    return int(getattr(cfg.detector, "batch_size", get_batch_size(cfg)))


def load_sequence_classifier_detector(cfg):
    detector_path_or_id = cfg.detector.detector_path_or_id
    model_cls = getattr(transformers, cfg.detector.model_class)
    tokenizer_cls = getattr(transformers, cfg.detector.tokenizer_class)
    tokenizer_name = getattr(cfg.detector, "tokenizer_name", detector_path_or_id)

    model = model_cls.from_pretrained(detector_path_or_id)
    tokenizer = tokenizer_cls.from_pretrained(tokenizer_name)

    model.eval()
    return DetectorModel(
        model,
        tokenizer,
        batch_size=get_batch_size(cfg),
        inverse_labels=getattr(cfg.detector, "inverse_labels", False),
    )


class BinocularsDetectorModel:
    def __init__(
        self,
        binoculars_detector,
        batch_size=8,
        score_threshold=0.8536432310785527,
    ):
        ensure_determinism()
        self.detector = binoculars_detector
        self.batch_size = batch_size
        self.score_threshold = float(score_threshold)

    def predict(self, dataset: Dataset):
        assert len(dataset["id"]) == len(set(dataset["id"])), "Duplicate IDS!"

        predicted_labels = []
        true_labels = []
        raw_scores = []
        scores = []

        texts = dataset["text"]
        labels = dataset["label"]

        for start in tqdm(
            range(0, len(texts), self.batch_size),
            desc="Processing Binoculars batches",
        ):
            stop = min(start + self.batch_size, len(texts))
            batch_texts = texts[start:stop]
            batch_scores = self.detector.compute_score(batch_texts)
            batch_scores = np.asarray(batch_scores, dtype=float)
            batch_normalized_scores = -batch_scores
            batch_preds = (np.asarray(batch_scores, dtype=float) < self.score_threshold).astype(int)

            predicted_labels.extend(batch_preds.tolist())
            true_labels.extend(labels[start:stop])
            raw_scores.extend(batch_scores.tolist())
            scores.extend(batch_normalized_scores.tolist())

        predictions = Predictions(
            predicted_labels=predicted_labels,
            true_labels=true_labels,
            scores=scores,
            raw_scores=raw_scores,
            default_threshold=-self.score_threshold,
            score_direction="lower_is_ai",
            scores_are_probabilities=False,
        )
        predictions.set_ids(list(dataset["id"]))
        return predictions


def load_binoculars_detector(cfg):
    from utils.binoculars import Binoculars

    detector = Binoculars(
        observer_name_or_path=cfg.detector.observer_name_or_path,
        performer_name_or_path=cfg.detector.performer_name_or_path,
        use_bfloat16=getattr(cfg.detector, "use_bfloat16", True),
        max_token_observed=getattr(cfg.detector, "max_token_observed", 512),
        mode=getattr(cfg.detector, "mode", "low-fpr"),
        trust_remote_code=getattr(cfg.detector, "trust_remote_code", False),
    )

    return BinocularsDetectorModel(
        detector,
        batch_size=get_binoculars_batch_size(cfg),
        score_threshold=getattr(cfg.detector, "score_threshold", detector.threshold),
    )


class FastDetectGPTDetectorModel:
    def __init__(
        self,
        detector,
        batch_size=1,
        score_threshold=0.0,
        inverse_labels=False,
    ):
        ensure_determinism()
        self.detector = detector
        self.batch_size = int(batch_size)
        self.score_threshold = float(score_threshold)
        self.inverse_labels = inverse_labels

    def predict(self, dataset: Dataset):
        assert len(dataset["id"]) == len(set(dataset["id"])), "Duplicate IDS!"

        predicted_labels = []
        true_labels = []
        scores = []
        raw_scores = []

        texts = dataset["text"]
        labels = dataset["label"]

        for start in tqdm(
            range(0, len(texts), self.batch_size),
            desc="Processing FastDetectGPT batches",
        ):
            stop = min(start + self.batch_size, len(texts))
            batch_texts = texts[start:stop]
            batch_scores = self.detector.compute_score(batch_texts)
            batch_scores = np.asarray(batch_scores, dtype=float)
            batch_preds = (np.asarray(batch_scores, dtype=float) >= self.score_threshold).astype(int)

            raw_scores.extend(batch_scores.tolist())
            scores.extend(batch_scores.tolist())
            predicted_labels.extend(batch_preds.tolist())
            true_labels.extend(labels[start:stop])

        predictions = Predictions(
            predicted_labels=predicted_labels,
            true_labels=true_labels,
            scores=scores,
            raw_scores=raw_scores,
            default_threshold=self.score_threshold,
            score_direction="higher_is_ai",
            scores_are_probabilities=False,
            metadata={"inverse_labels": bool(self.inverse_labels)},
        )
        predictions.set_ids(list(dataset["id"]))
        return predictions


def load_fastdetectgpt_detector(cfg):
    from utils.fastdetectgpt import FastDetectGPT

    detector = FastDetectGPT(
        sampling_model_name_or_path=cfg.detector.sampling_model_name_or_path,
        scoring_model_name_or_path=cfg.detector.scoring_model_name_or_path,
        max_token_observed=getattr(cfg.detector, "max_token_observed", 512),
        dtype=getattr(cfg.detector, "dtype", "bf16"),
        quantization=getattr(cfg.detector, "quantization", "none"),
        trust_remote_code=getattr(cfg.detector, "trust_remote_code", False),
        device_map=getattr(cfg.detector, "device_map", "auto"),
        local_files_only=getattr(cfg.detector, "local_files_only", False),
    )

    return FastDetectGPTDetectorModel(
        detector,
        batch_size=get_fastdetectgpt_batch_size(cfg),
        score_threshold=getattr(cfg.detector, "score_threshold", 0.0),
        inverse_labels=getattr(cfg.detector, "inverse_labels", False),
    )



class DetectorModel:
    def __init__(self, model, tokenizer, batch_size=16, inverse_labels=False):
        ensure_determinism()
        self.errors = 0

        self.model = model
        self.model.to(get_device())
        self.tokenizer = tokenizer
        # self.optimal_threshold = 0.5  # Default threshold for binary classification
        self.batch_size = batch_size
        self.negate = False  # For RADAR model, we need to negate predictions
        if inverse_labels:
            self.enable_negate()

    def enable_negate(self):
        """
        Set whether to negate predictions (for RADAR model).
        """
        print("Enable negate for predictions (for RADAR model)")
        self.negate = True

    def get_tokenized_dataset(self, dataset):
        """
        Tokenizes dataset with padding & truncation, ensuring uniform tensor lengths.
        """
        # label_map = {"human" : constants.LABEL_HUMAN, "machine": constants.LABEL_MACHINE}
        # print(dataset)
        def tokenize_function(examples):

            tokenized_inputs = self.tokenizer(examples["text"], 
                            padding=True, 
                            truncation=True, 
                            max_length=512,
                            return_tensors="pt")  # Return PyTorch tensors
        
            # tokenized_inputs["label"] = [label_map[label] for label in examples["label"]]
            #if "label" in examples:
            #   tokenized_inputs["label"] = [label_map.get(label, 0) for label in examples["label"]]

            return tokenized_inputs
        tokenized_dataset = dataset.map(tokenize_function, batched=True, load_from_cache_file=False)
        
        # Remove unnecessary columns
        #tokenized_dataset = tokenized_dataset.remove_columns(["text", "id", "label"])
        tokenized_dataset = tokenized_dataset.remove_columns(["text", "id"])
        return tokenized_dataset

    def get_data_loader(self, dataset):
        ## Remove text,id
        tokenized_dataset = self.get_tokenized_dataset(dataset)
        test_dataset = tokenized_dataset  # Assume train dataset is used
        data_collator = DataCollatorWithPadding(self.tokenizer)
        data_loader = DataLoader(test_dataset, batch_size=self.batch_size, collate_fn=data_collator)
        return data_loader

    def predict(self, dataset):
        """
        Runs inference on the dataset and returns predictions, labels, and probabilities.
        """
        device = get_device()
        assert len(dataset['id']) == len(set(dataset['id'])), "Duplicate IDS!"
        data_loader = self.get_data_loader(dataset)
        predictions = predict_batch(data_loader, self.model, device, self.negate)
        ids = [id for id in dataset['id']]
        predictions.set_ids(ids)
        return predictions


def load_detector_model(cfg):
    detector_type = getattr(cfg.detector, "detector_type", "sequence_classifier")
    if detector_type == "binoculars":
        return load_binoculars_detector(cfg)
    if detector_type == "fastdetectgpt":
        return load_fastdetectgpt_detector(cfg)
    return load_sequence_classifier_detector(cfg)

# class RadarModel(DetectorModel):
#     def __init__(self):
#         detector_path_or_id = "TrustSafeAI/RADAR-Vicuna-7B"
#         detector = transformers.AutoModelForSequenceClassification.from_pretrained(detector_path_or_id)
#         tokenizer = transformers.AutoTokenizer.from_pretrained(detector_path_or_id)
#         detector.to(get_device())
#         detector.eval()
#         super().__init__(detector, tokenizer)
#         self.optimal_threshold = 0.05  # Optimal threshold for RADAR model
#         self.enable_negate()

# class BertModel(DetectorModel):
#     def __init__(self):
#         model = BertForSequenceClassification.from_pretrained(
#             os.path.join(constants.PRETRAINED_PATH, "BERT-Defense")
#         )
#         tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
#         model.eval()
#         super().__init__(model, tokenizer)

class RoBertaDefenseModel(DetectorModel):
    def __init__(self):
        model_path = os.path.expanduser(
            os.path.join("~", ".pretrained-models", "RoBERTa-Defense")
        )


        model = RobertaForSequenceClassification.from_pretrained(model_path)
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        model.eval()
        super().__init__(model, tokenizer)
