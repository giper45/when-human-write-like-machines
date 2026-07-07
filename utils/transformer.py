import torch

@torch.inference_mode()
def generate_text(
    tokenizer,
    model,
    user_prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    max_new_tokens: int = 256,
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

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()