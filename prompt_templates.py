PROMPT_TEMPLATES = {
    "completion": "{prompt}",
    "llama-3-instruct": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "llama-2-instruct": "[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]",
    "mistral-instruct": "[INST] {system_prompt} {prompt} [/INST]",
}

MODELS = {
    "meta-llama-3-8b": PROMPT_TEMPLATES["completion"],
    "meta-llama-3-8b-instruct": PROMPT_TEMPLATES["llama-3-instruct"],
    "meta-llama-3-70b": PROMPT_TEMPLATES["completion"],
    "meta-llama-3-70b-instruct": PROMPT_TEMPLATES["llama-3-instruct"],

    "llama-2-7b": PROMPT_TEMPLATES["completion"],
    "llama-2-7b-chat-instruct": PROMPT_TEMPLATES["llama-2-instruct"],
    "llama-2-13b": PROMPT_TEMPLATES["completion"],
    "llama-2-13b-chat-instruct": PROMPT_TEMPLATES["llama-2-instruct"],
    "llama-2-70b": PROMPT_TEMPLATES["completion"],
    "llama-2-70b-chat-instruct": PROMPT_TEMPLATES["llama-2-instruct"],

    "mistral-7b-v0.1": PROMPT_TEMPLATES["completion"],
    "mistral-7b-instruct-v0.1": PROMPT_TEMPLATES["mistral-instruct"],
    "mistral-7b-instruct-v0.2": PROMPT_TEMPLATES["mistral-instruct"],
    "mistral-8x7b-instruct-v0.1": PROMPT_TEMPLATES["mistral-instruct"]
}


def get_prompt_template(model_id: str) -> str:
    if model_id in MODELS:
        return MODELS[model_id]
    else:
        return "{prompt}"