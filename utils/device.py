import gc
import torch
from transformers import BitsAndBytesConfig
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Cleared GPU memory")



def clear_gpu_full():
    # elimina riferimenti noti se esistono
    for name in ["model", "tokenizer", "inputs", "output_ids", "generated_ids"]:
        if name in globals():
            del globals()[name]

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

    print("GPU cleanup attempted")
    if torch.cuda.is_available():
        print("allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
        print("reserved:", torch.cuda.memory_reserved() / 1024**2, "MB")

def get_quantization_config():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,                 # usa 4-bit quantization
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    return quant_config

import torch

# def find_max_batch_size(filter_fn, texts, start=1, end=128):
#     """
#     Finds the largest batch size that does not cause CUDA OOM.

#     Args:
#         filter_fn: function that takes a batch {"text": [...]} and returns results
#         texts: list of texts to test with
#         start: minimum batch size to try
#         end: maximum batch size to try (upper search bound)

#     Returns:
#         max_safe_batch: int
#     """
#     low, high = start, end
#     max_safe = start

#     while low <= high:
#         print(f"Trying batch size: {(low + high) // 2}")
#         mid = (low + high) // 2
#         batch = {"text": texts[:mid]}  # take a slice for testing
#         try:
#             # clear cache before attempt
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
            
#             _ = filter_fn(batch)

#             # if success, move up
#             max_safe = mid
#             low = mid + 1
#         except RuntimeError as e:
#             if "out of memory" in str(e).lower():
#                 # if OOM, move down
#                 if torch.cuda.is_available():
#                     torch.cuda.empty_cache()
#                 high = mid - 1
#             else:
#                 raise e  # unexpected error -> re-raise

#     return max_safe

# def get_stanza_pipeline(lang='en'):
#     import stanza
#     if torch.cuda.is_available():
#         return stanza.Pipeline(lang=lang, processors='tokenize, pos, lemma', use_gpu=True)
#     else:
#         return stanza.Pipeline(lang=lang, processors='tokenize, pos, lemma', use_gpu=False)