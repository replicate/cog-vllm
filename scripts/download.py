# downloads and save it on the dir to get pushed to replicate.


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-chat-hf", torch_dtype=torch.float16
)
model.save_pretrained("./model_path/meta-llama/Llama-2-13b-chat-hf")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
tokenizer.save_pretrained("./model_path/meta-llama/Llama-2-13b-chat-hf")
