import torch
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def clean_text(text: str) -> str:
    """
    Cleans the generated text by removing unwanted characters and formatting.
    """
    # Remove leading/trailing whitespace
    text = text.strip()
    # Replace multiple spaces with a single space
    text = ' '.join(text.split())
    # Remove newlines
    text = text.replace('\n', ' ')
    return text

@torch.inference_mode()
def generate_text(
    tokenizer,
    model,
    user_prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    max_new_tokens: int = 512,
):
        # Gemma chat template does not support sy"stem role.
    model_id = model.config._name_or_path
    if "gemma" in model_id.lower():
        print("GEMMA MODEL")
        merged_prompt = f"""Instruction:
{system_prompt}

Task:
{user_prompt}

Return only the requested output."""
        messages = [
            {"role": "user", "content": merged_prompt}
        ]


    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    input_length = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][input_length:]

    return clean_text(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_model_bf16(model_id: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    model.eval()
    return model, tokenizer
