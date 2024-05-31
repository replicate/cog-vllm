COMPLETION = "<s>{prompt}"
LLAMA_3_INSTRUCT = "<s><|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
LLAMA_2_INSTRUCT = "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
MISTRAL_INSTRUCT = "<s>[INST] {system_prompt} {prompt} [/INST]"
