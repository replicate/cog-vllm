from vllm import LLM, SamplingParams

import shutil
import time
from typing import List
import zipfile

import torch


from cog import BasePredictor, ConcatenateIterator, Input, Path
from config import DEFAULT_MODEL_NAME, load_tokenizer, load_tensorizer, pull_gcp_file
from subclass import YieldingLlama
from peft import PeftModel
import os


class Predictor(BasePredictor):
    def __init__(self):
        self.engine = LLM(model="meta-llama/Llama-2-13b-chat-hf")

    def predict(
        self,
        prompt: str = Input(description=f"Prompt to send to Llama v2."),
        num_samples: int = Input(
            description=f"Number of samples to generate.", default=1, ge=1, le=10
        ),
        max_length: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            ge=1,
            default=500,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
            ge=0.01,
            le=5,
            default=0.5,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.01,
            le=1.0,
            default=1.0,
        ),
        repetition_penalty: float = Input(
            description="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it.",
            ge=0.01,
            le=5,
            default=1,
        ),
        # debug: bool = Input(
        #     description="provide debugging output in logs", default=False
        # )
    ) -> List[str]:
        prompt = "User: " + prompt + "\nAssistant: "

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=repetition_penalty,
            max_tokens=max_length,
        )

        outputs = self.engine.generate([prompt] * num_samples, sampling_params)
        # Print the outputs.
        gen_texts = []
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            gen_texts.append(generated_text)

        return gen_texts