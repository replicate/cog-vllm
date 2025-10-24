#!/bin/sh

# Mirror of https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
COG_WEIGHTS="https://weights.replicate.delivery/default/Qwen/Qwen3-VL-8B-Instruct/model.tar"

exec cog predict \
    -e COG_WEIGHTS=$COG_WEIGHTS \
    -i prompt="Who are you?" \
    -i max_tokens=512 \
    -i temperature=0.6 \
    -i top_p=0.9 \
    -i top_k=50 \
    -i presence_penalty=0.0 \
    -i frequency_penalty=0.0 \
    -i prompt_template="<s>[INST] {prompt} [/INST] "
