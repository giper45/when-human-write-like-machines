import json
import os
import random
import re
import time
import torch
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from transformers import RobertaTokenizer, DataCollatorWithPadding
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, RobertaTokenizer, BertTokenizer

from torch.utils.data import DataLoader
import transformers
import numpy as np
from os.path import join as j

from utils import constants
from utils.device import get_device
# from utils.evaluatuon.evaluation import Predictions

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




def predict_batch(data_loader, model, device, should_negate):
    """
    Runs batch inference using a DataLoader.
    """
    predicted_labels = []
    true_labels = []
    pred_probs = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Processing batches"):
            batch = {key: val.to(device) for key, val in batch.items()}  # Move batch to device
            labels = batch['labels'].cpu().tolist() 
            inputs = {key: batch[key] for key in batch if key != 'labels'}
            try:
                outputs = model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1).cpu().tolist()  # Get predicted class labels
                probs = torch.nn.functional.softmax(logits, dim=-1)[:, 1].cpu().tolist()  # Get probability of class 1

            except RuntimeError as e:
                print(f"CUDA ERROR: {e}")
                print(f"Batch keys: {batch.keys()}")
                print(f"Input shapes: {[batch[key].shape for key in batch]}")
                raise e  # Force stop to debug


            predicted_labels.extend(preds)
            true_labels.extend(labels)
            pred_probs.extend(probs)
    return Predictions(predicted_labels, true_labels, pred_probs, should_negate)



class DetectorModel:
    def __init__(self, model, tokenizer, batch_size=16):
        ensure_determinism()
        self.errors = 0

        self.model = model
        self.model.to(get_device())
        self.tokenizer = tokenizer
        # self.optimal_threshold = 0.5  # Default threshold for binary classification
        self.batch_size = batch_size
        self.negate = False  # For RADAR model, we need to negate predictions

    def enable_negate(self):
        """
        Set whether to negate predictions (for RADAR model).
        """
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
