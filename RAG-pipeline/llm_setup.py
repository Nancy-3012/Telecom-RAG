import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Plain transformers + torch -- both already confirmed working on this machine.
# Deliberately avoids llama-cpp-python (needs a working C++ compiler toolchain)
# and bitsandbytes (CUDA-only, won't run on a Mac). Qwen2.5-3B is small enough
# to run directly with no quantization library needed.
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

_pipe = None


def load_llm():
    global _pipe
    if _pipe is not None:
        return _pipe

    print(f"Loading tokenizer and model: {MODEL_NAME} (first run downloads ~6GB)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cpu"
        )
    except Exception:
        print("bfloat16 not supported here, falling back to float32...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32, device_map="cpu"
        )

    _pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=400,
        temperature=0.3,
        do_sample=True,
    )
    print("LLM ready.")
    return _pipe


def generate(prompt, max_tokens=400, temperature=0.3):
    """
    Formats the prompt using the model's own chat template (handled by the
    tokenizer, so we don't have to guess the exact instruction format) and
    generates a response.
    """
    pipe = load_llm()
    messages = [{"role": "user", "content": prompt}]
    formatted = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    output = pipe(formatted, max_new_tokens=max_tokens, temperature=temperature, do_sample=True)
    full_text = output[0]["generated_text"]
    return full_text[len(formatted):].strip()


if __name__ == "__main__":
    response = generate("What is 5G NR handover? Answer in one sentence.")
    print("Response:", response)