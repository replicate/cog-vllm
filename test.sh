#!/bin/sh
# python3 setup.py -m codegemma-2b -w https://weights.replicate.delivery/default/official-models/hf/google/codegemma-2b -t codegemma-2b
cog predict -e COG_WEIGHTS=https://replicate.delivery/czjl/YQem14A8LwQUfEHmei2ahogNWAXXjsqUfcNcgxn9lqe533PXC/model.tar \
           -i prompt="write a python program that prints Hello World!" \
           -i max_new_tokens=512 \
           -i temperature=0.6 \
           -i top_p=0.9 \
           -i top_k=50 \
           -i presence_penalty=0.0 \
           -i frequency_penalty=0.0 \
           -i prompt_template="<s>[INST] {prompt} [/INST] "
