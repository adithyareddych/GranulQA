import torch
from functools import lru_cache
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)

_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
_MAX_NEW_TOKENS = 256

# ── 4‑bit quantisation config ───────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading Llama-2-7B-chat in 4-bit …")
# 1) single tokenizer (we can share it)
tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME, use_fast=True)

# 2) load the quantized model once
model = AutoModelForCausalLM.from_pretrained(
    _MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

@lru_cache(maxsize=1)
def get_pipeline(task: str = "text-generation"):
    """
    Reuse the 4‑bit model and tokenizer in a HF pipeline.
    """
    return pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=_MAX_NEW_TOKENS,
    )

def generate(prompt: str, temperature: float = 0.1) -> str:
    """
    Generate text with the quantized LLaMA‑2 7B.
    """
    pipe = get_pipeline("text-generation")
    out  = pipe(prompt, temperature=temperature, return_full_text=False)
    return out[0]["generated_text"]
