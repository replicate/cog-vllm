import time
from uuid import uuid4

import torch
from cog import BasePredictor, ConcatenateIterator, Input
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from typing import NamedTuple

import prompt_templates
from utils import download_and_extract_tarball, remove_system_prompt_input

PROMPT_TEMPLATE = prompt_templates.COMPLETION  # Change this for instruct models

SYSTEM_PROMPT = "You are a helpful assistant."


class PredictorConfig(NamedTuple):
    prompt_template: str


class Predictor(BasePredictor):
    async def setup(self, weights: str):  # pylint: disable=invalid-overridden-method, signature-differs
        if not weights:
            raise ValueError(
                "Weights must be provided. "
                "Set COG_WEIGHTS environment variable to "
                "a URL to a tarball containing the weights file "
                "or a path to the weights file."
            )

        weights = await download_and_extract_tarball(str(weights))

        engine_args = AsyncEngineArgs(
            dtype="auto",
            tensor_parallel_size=max(torch.cuda.device_count(), 1),
            model=weights,
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)  # pylint: disable=attribute-defined-outside-init
        self.tokenizer = (  # pylint: disable=attribute-defined-outside-init
            self.engine.engine.tokenizer.tokenizer
            if hasattr(self.engine.engine.tokenizer, "tokenizer")
            else self.engine.engine.tokenizer
        )

    async def predict(  # pylint: disable=invalid-overridden-method, arguments-differ
        self,
        prompt: str = Input(description="Prompt", default=""),
        system_prompt: str = Input(
            description="System prompt to send to the model. This is prepended to the prompt and helps guide system behavior.",
            default=SYSTEM_PROMPT,
        ),
        min_tokens: int = Input(
            description="The minimum number of tokens the model should generate as output.",
            default=0,
        ),
        max_tokens: int = Input(
            description="The maximum number of tokens the model should generate as output.",
            default=512,
        ),
        temperature: float = Input(
            description="The value used to modulate the next token probabilities.",
            default=0.6,
        ),
        top_p: float = Input(
            description="A probability threshold for generating the output. If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751).",
            default=0.9,
        ),
        top_k: int = Input(
            description="The number of highest probability tokens to consider for generating the output. If > 0, only keep the top k tokens with highest probability (top-k filtering).",
            default=50,
        ),
        presence_penalty: float = Input(description="Presence penalty", default=0.0),
        frequency_penalty: float = Input(description="Frequency penalty", default=0.0),
        stop_sequences: str = Input(
            description="A comma-separated list of sequences to stop generation at. For example, '<end>,<stop>' will stop generation at the first instance of 'end' or '<stop>'.",
            default=None,
        ),
        prompt_template: str = Input(
            description="Prompt template. The string `{prompt}` will be substituted for the input prompt. If you want to generate dialog output, use this template as a starting point and construct the prompt string manually, leaving `prompt_template={prompt}`.",
            default=PROMPT_TEMPLATE,
        ),
    ) -> ConcatenateIterator[str]:
        start = time.time()

        prompt = prompt_template.format(prompt=prompt, system_prompt=system_prompt)

        sampling_params = SamplingParams(
            n=1,
            top_k=(-1 if (top_k or 0) == 0 else top_k),
            top_p=top_p,
            temperature=temperature,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            stop_token_ids=[self.tokenizer.eos_token_id],
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            use_beam_search=False,
        )
        if isinstance(stop_sequences, str) and stop_sequences:
            sampling_params.stop = stop_sequences.split(",")
        else:
            sampling_params.stop = (
                list(stop_sequences) if isinstance(stop_sequences, list) else []
            )

        generator = self.engine.generate(prompt, sampling_params, uuid4().hex)
        start = 0

        async for result in generator:
            assert (
                len(result.outputs) == 1
            ), "Expected exactly one output from generation request."

            text = result.outputs[0].text

            # Normalize text by removing any incomplete surrogate pairs (common with emojis)
            text = text.replace("\N{REPLACEMENT CHARACTER}", "")

            yield text[start:]

            start = len(text)

        print(f"Generation took {time.time() - start:.2f}s")
        print(f"Formatted prompt: {prompt}")

    if "system_prompt" not in PROMPT_TEMPLATE:
        predict = remove_system_prompt_input(predict)
