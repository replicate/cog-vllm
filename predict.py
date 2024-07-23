import json
import os
import time
from typing import NamedTuple, Optional
from uuid import uuid4

import jinja2
#import torch
from cog import BasePredictor, ConcatenateIterator, Input
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams

import prompt_templates
from utils import resolve_model_path

PROMPT_TEMPLATE = prompt_templates.LLAMA_3_INSTRUCT

SYSTEM_PROMPT = "You are a helpful assistant."


class PredictorConfig(NamedTuple):
    prompt_template: Optional[str] = None


class UserError(Exception):
    pass


class TritonError(Exception):
    pass


def format_prompt(
    prompt: str, prompt_template: str, system_prompt: Optional[str]
) -> str:
    if not prompt_template:
        prompt_template = "{prompt}"
    if prompt and "{prompt}" not in prompt_template:
        raise UserError(
            "E1003 BadPromptTemplate: You have submitted both a prompt and a prompt template that doesn't include '{prompt}'."
            "Your prompt would not be used. "
            "If don't want to use formatting, use your full prompt for the prompt argument and set prompt_template to '{prompt}'."
        )
    try:
        return prompt_template.format(
            system_prompt=system_prompt or "",
            prompt=prompt,
        )
    except (ValueError, KeyError, IndexError) as e:
        # sometimes people put the prompt in prompt_template
        if len(prompt_template) > len(prompt):
            raise UserError(
                "E1004 PromptTemplateError: Prompt template must be a valid python format spec. "
                "Did you submit your prompt as `prompt_template` instead of `prompt`? "
                'If you want finer control over templating, set prompt_template to `"{prompt}"` to disable formatting. '
                "You can't put JSON in prompt_template, because braces will be parsed as a python format string. "
                f"Detail: {repr(e)}"
            )
        # most common case is "unmatched '{' in format spec",
        # but IndexError/KeyError and other formatting errors can happen
        # str(KeyError) is only the missing key which can be confusing
        raise UserError(
            f"E1004 PromptTemplateError: Prompt template must be a valid python format spec: {repr(e)}"
        )
    
class Predictor(BasePredictor):
    async def setup(self):
        weights = "https://weights.replicate.delivery/default/hf/meta-llama/llama-3.1-405b-instruct-fp8-revised3.tar"

        weights = await resolve_model_path(str(weights))

        if os.path.exists(os.path.join(weights, "predictor_config.json")):
            print("Loading predictor_config.json")
            with open(
                os.path.join(weights, "predictor_config.json"), "r", encoding="utf-8"
            ) as f:
                config = json.load(f)
            self.config = PredictorConfig(
                **config
            )  # pylint: disable=attribute-defined-outside-init

        else:
            self.config = PredictorConfig()

        print("Predictor Configuration:")
        for key, value in self.config._asdict().items():
            print(f"{key}: {value}")

        engine_args = AsyncEngineArgs(
            dtype="auto",
            quantization="fbgemm_fp8",
            tensor_parallel_size=8, #max(torch.cuda.device_count(), 1),
            model=weights,
            #gpu_memory_utilization=0.8,
            max_model_len=4096,
        )

        self.engine = AsyncLLMEngine.from_engine_args(
            engine_args
        )  # pylint: disable=attribute-defined-outside-init

        self.tokenizer = (
            self.engine.engine.tokenizer.tokenizer
            if hasattr(self.engine.engine.tokenizer, "tokenizer")
            else self.engine.engine.tokenizer
        )  # pylint: disable=attribute-defined-outside-init

        if self.config.prompt_template:
            print(
                f"Using prompt template from `predictor_config.json`: {self.config.prompt_template}"
            )
            self.tokenizer.chat_template = self.config.prompt_template

        elif self.tokenizer.chat_template:
            print(
                f"Using prompt template from `tokenizer`: {self.tokenizer.chat_template}"
            )
        else:
            print(
                f"No prompt template specified in `predictor_config.json` or `tokenizer`, defaulting to: {PROMPT_TEMPLATE}"
            )
            self.tokenizer.chat_template = PROMPT_TEMPLATE

    async def predict(  # pylint: disable=invalid-overridden-method, arguments-differ
        self,
        prompt: str = Input(description="Prompt", default=""),
        system_prompt: str = Input(
            description="System prompt to send to the model. This is prepended to the prompt and helps guide system behavior. Ignored for non-chat models.",
            default="You are a helpful assistant.",
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
            description="A template to format the prompt with. If not provided, the default prompt template will be used.",
            default=None,
        ),
    ) -> ConcatenateIterator[str]:
        start = time.time()

        if prompt_template:
            prompt = format_prompt(
                prompt=prompt,
                prompt_template=prompt_template,
                system_prompt=system_prompt,
            )

        elif self.tokenizer.chat_template:
            system_prompt = "" if system_prompt is None else system_prompt
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except jinja2.exceptions.TemplateError:
                messages = [
                    {"role": "user", "content": "\n\n".join([system_prompt, prompt])}
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        elif system_prompt:
            self.log(
                "Warning: ignoring system prompt because no chat template was configured"
            )

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

        request_id = uuid4().hex
        generator = self.engine.generate(prompt, sampling_params, request_id)
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

        self.log(f"Generation took {time.time() - start:.2f}s")
        self.log(f"Formatted prompt: {prompt}")
