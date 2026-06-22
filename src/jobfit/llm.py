from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

_tokenizer = None
_model = None

def _load_model():
    """Lazy loader — model is only loaded on first generate_text() call."""
    global _tokenizer, _model
    if _model is None:
        print(f"Loading {MODEL_NAME}...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16,      
            low_cpu_mem_usage=True     
)
        _model.eval()
        print("Model ready.")
    return _tokenizer, _model


def generate_text(prompt, temperature=0.7, top_k=50, top_p=0.9, max_new_tokens=200):
    tokenizer, model = _load_model()

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(formatted, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id  # avoids padding warning
        )

    new_tokens = outputs[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)