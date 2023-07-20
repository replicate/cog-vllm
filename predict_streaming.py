from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams

import shutil
import time
from typing import List
import zipfile

import torch
from cog import BasePredictor, ConcatenateIterator, Input, Path
from peft import PeftModel
import os

import asyncio


DEFAULT_PROMPT = """[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>\n<<INSTRUCTION>> [/INST]"""


class Predictor(BasePredictor):
    def setup(self):
        args = AsyncEngineArgs(
            model="./model_path/meta-llama/Llama-2-13b-chat-hf",
            dtype="float16",
            max_num_seqs=4096,
        )
        self.engine = AsyncLLMEngine.from_engine_args(args)
        self.tokenizer = self.engine.engine.tokenizer

    async def generate_stream(
        self,
        prompt,
        temperature=1.0,
        top_p=1.0,
        max_new_tokens=4,
        stop_str=None,
        stop_token_ids=None,
        repetition_penalty=1.0,
        echo=True,
    ):
        context = prompt
        stop_token_ids = stop_token_ids or []
        stop_token_ids.append(self.tokenizer.eos_token_id)

        if isinstance(stop_str, str) and stop_str != "":
            stop = [stop_str]
        elif isinstance(stop_str, list) and stop_str != []:
            stop = stop_str
        else:
            stop = []

        for tid in stop_token_ids:
            stop.append(self.tokenizer.decode(tid))

        # make sampling params in vllm
        top_p = max(top_p, 1e-5)
        if temperature <= 1e-5:
            top_p = 1.0
        sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            top_p=top_p,
            use_beam_search=False,
            stop=stop,
            max_tokens=max_new_tokens,
            frequency_penalty=repetition_penalty,
        )
        results_generator = self.engine.generate(context, sampling_params, 0)

        async for request_output in results_generator:
            prompt = request_output.prompt
            yield request_output.outputs[-1].text

    def predict(
        self,
        prompt: str = Input(description=f"Prompt to send to Llama v2."),
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
        prompt_template: str = Input(
            description="Template for the prompt to send to Llama v2. In this format, <<INSTRUCTION>> will get replaced by the input prompt (first argument of this api)",
            default=DEFAULT_PROMPT,
        ),
    ) -> ConcatenateIterator[str]:
        prompt_templated = prompt_template.replace("<<INSTRUCTION>>", prompt)

        loop = asyncio.get_event_loop()
        gen = self.generate_stream(
            prompt_templated,
            temperature,
            top_p,
            max_length,
            repetition_penalty=repetition_penalty,
        )
        prv_value = ""
        value = ""
        while True:
            prv_value = value
            try:
                value = loop.run_until_complete(gen.__anext__())
                yield value[len(prv_value) :]
            except StopAsyncIteration:
                break
