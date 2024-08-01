# pylint: disable=missing-module-docstring, no-name-in-module, attribute-defined-outside-init
import json
import os
import time
from typing import Optional, Dict
from uuid import uuid4
from dataclasses import dataclass, field
from pprint import pprint
import inspect

import jinja2
import torch  # pylint: disable=import-error
from cog import BasePredictor, ConcatenateIterator, Input
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs  # pylint: disable=import-error
from vllm.sampling_params import SamplingParams  # pylint: disable=import-error

import prompt_templates
from utils import resolve_model_path

PROMPT_TEMPLATE = prompt_templates.COMPLETION  # Change this for instruct models

SYSTEM_PROMPT = "You are a helpful assistant."


@dataclass
class PredictorConfig:
    """
    PredictorConfig is a configuration class for the Predictor.

    Attributes:
        prompt_template (Optional[str]): A template to format the prompt with. If not provided,
                                         the default prompt template will be used.
        engine_args (Optional[Dict]): A dictionary of engine arguments. If not provided,
                                      an empty dictionary will be used.
    """

    prompt_template: Optional[str] = None
    engine_args: Optional[Dict] = field(default_factory=dict)

    def __post_init__(self):
        if self.engine_args is None:
            self.engine_args = {}
        if not isinstance(self.engine_args, dict):
            raise UserError(
                "E1202 InvalidPredictorConfig: engine_args must be "
                "a valid JSON object that maps to a dictionary."
            )


# pylint: disable=missing-class-docstring
class UserError(Exception):
    pass


# pylint: disable=missing-class-docstring
class VLLMError(Exception):
    pass


def format_prompt(
    prompt: str, prompt_template: str, system_prompt: Optional[str]
) -> str:
    """
    Formats the given prompt using the provided prompt template and system prompt.

    Args:
        prompt (str): The user-provided prompt to be formatted.
        prompt_template (str): The template string that includes placeholders for the prompt
        and, optionally, system prompt. Must include {prompt}.
        system_prompt (Optional[str]): An optional system prompt to be included in the
        formatted prompt.

    Returns:
        str: The formatted prompt string.

    Raises:
        UserError: If the prompt template does not include the '{prompt}' placeholder or if
        there is an error in formatting.
    """
    if not prompt_template:
        prompt_template = "{prompt}"
    if prompt and "{prompt}" not in prompt_template:
        raise UserError(
            "E1003 BadPromptTemplate: You have submitted both a prompt and a "
            "prompt template that doesn't include '{prompt}'. Your prompt would "
            "not be used. If don't want to use formatting, use your full prompt "
            "for the prompt argument and set prompt_template to '{prompt}'."
        )
    try:
        return prompt_template.format(system_prompt=system_prompt or "", prompt=prompt)
    except (ValueError, KeyError, IndexError) as e:
        # sometimes people put the prompt in prompt_template
        if len(prompt_template) > len(prompt):
            raise UserError(
                "E1004 PromptTemplateError: Prompt template must be a valid "
                "python format spec. Did you submit your prompt as "
                "`prompt_template` instead of `prompt`? If you want finer "
                'control over templating, set prompt_template to `"{prompt}"` '
                "to disable formatting. You can't put JSON in prompt_template, "
                "because braces will be parsed as a python format string. "
                f"Detail: {repr(e)}"
            ) from e
        # most common case is "unmatched '{' in format spec",
        # but IndexError/KeyError and other formatting errors can happen
        # str(KeyError) is only the missing key which can be confusing
        raise UserError(
            f"E1004 PromptTemplateError: Prompt template must be a valid "
            f"python format spec: {repr(e)}"
        ) from e


# pylint: disable=missing-class-docstring
class Predictor(BasePredictor):
    async def setup(
        self, weights: str
    ):  # pylint: disable=invalid-overridden-method, signature-differs
        if not weights:
            raise ValueError(
                "Weights must be provided. "
                "Set COG_WEIGHTS environment variable to "
                "a URL to a tarball containing the weights file "
                "or a path to the weights file."
            )

        weights = await resolve_model_path(str(weights))
        self.config = self.load_config(weights)

        engine_args = self.config.engine_args or {}
        engine_args["model"] = weights
        if "dtype" not in engine_args:
            engine_args["dtype"] = "auto"
        if "tensor_parallel_size" not in engine_args:
            engine_args["tensor_parallel_size"] = max(torch.cuda.device_count(), 1)

        engine_args = AsyncEngineArgs(**engine_args)

        try:
            # pylint: disable=attribute-defined-outside-init
            self.engine = AsyncLLMEngine.from_engine_args(
                engine_args
            )  # pylint: disable=attribute-defined-outside-init
        except TypeError as e:
            print(f"E1201 UnexpectedEngineArg: {e}")
            raise
        except Exception as e:
            print(f"E1200 VLLMUnknownError: {e}")
            raise

        # pylint: disable=attribute-defined-outside-init
        self.tokenizer = (
            self.engine.engine.tokenizer.tokenizer
            if hasattr(self.engine.engine.tokenizer, "tokenizer")
            else self.engine.engine.tokenizer
        )

        if self.config.prompt_template:
            print(
                f"Using prompt template from `predictor_config.json`: {self.config.prompt_template}"
            )
            self.tokenizer.chat_template = self.config.prompt_template
            self.prompt_template = None

        elif self.tokenizer.chat_template:
            print(
                f"Using prompt template from `tokenizer`: {self.tokenizer.chat_template}"
            )
            self.prompt_template = None
        else:
            print(
                "No prompt template specified in `predictor_config.json` or "
                f"`tokenizer`, defaulting to: {PROMPT_TEMPLATE}"
            )
            self.tokenizer.chat_template = None
            self.prompt_template = PROMPT_TEMPLATE

        self._testing = True
        generator = self.predict(
            **dict(self._defaults, **{"max_tokens": 3, "prompt": "hi"})
        )
        test_output = "".join([tok async for tok in generator])
        print("Test prediction output:", test_output)
        self._testing = False

    async def predict(  # pylint: disable=invalid-overridden-method, arguments-differ, too-many-arguments, too-many-locals
        self,
        prompt: str = Input(description="Prompt", default=""),
        system_prompt: str = Input(
            description="System prompt to send to the model. This is prepended to "
            "the prompt and helps guide system behavior. Ignored for non-chat models.",
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
            description="A probability threshold for generating the output. If < 1.0, only keep "
            "the top tokens with cumulative probability >= top_p (nucleus filtering). "
            "Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751).",
            default=0.9,
        ),
        top_k: int = Input(
            description="The number of highest probability tokens to consider for generating "
            "the output. If > 0, only keep the top k tokens with highest probability "
            "(top-k filtering).",
            default=50,
        ),
        presence_penalty: float = Input(description="Presence penalty", default=0.0),
        frequency_penalty: float = Input(description="Frequency penalty", default=0.0),
        stop_sequences: str = Input(
            description="A comma-separated list of sequences to stop generation at. "
            "For example, '<end>,<stop>' will stop generation at the first instance of "
            "'end' or '<stop>'.",
            default=None,
        ),
        prompt_template: str = Input(
            description="A template to format the prompt with. If not provided, "
            "the default prompt template will be used.",
            default=None,
        ),
    ) -> ConcatenateIterator[str]:
        start = time.time()

        if prompt_template or self.prompt_template:
            prompt_template = prompt_template or self.prompt_template
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
            # pylint: disable=no-member
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

        # pylint: disable=no-member
        self.log(f"Generation took {time.time() - start:.2f}s")
        self.log(f"Formatted prompt: {prompt}")

    def load_config(self, weights: str) -> PredictorConfig:
        """
        Load the predictor configuration from the specified weights directory or
        the current directory.

        Load `predictor_config.json` from the weights directory or current directory.
        Return a default PredictorConfig object if not found or an error occurs.

        Priority:
        1. Load `predictor_config.json` from the specified weights directory.
        2. If not found, load `predictor_config.json` from the current directory.
        3. If not found or an error occurs, return a default PredictorConfig object.

        Args:
            weights (str): The path to the weights directory.

        Returns:
            PredictorConfig: The loaded predictor configuration.
        """
        if os.path.exists(os.path.join(weights, "predictor_config.json")):
            predictor_config_path = os.path.join(weights, "predictor_config.json")
        elif os.path.exists("./predictor_config.json"):
            predictor_config_path = "./predictor_config.json"
        else:
            predictor_config_path = None
        if predictor_config_path:
            try:
                print("Loading predictor_config.json")
                with open(
                    predictor_config_path,
                    "r",
                    encoding="utf-8",
                ) as f:
                    config = json.load(f)
                # pylint: disable=attribute-defined-outside-init
                config = PredictorConfig(**config)
            except Exception as e:
                raise UserError(f"E1202 InvalidPredictorConfig: {e}") from e

        else:
            config = PredictorConfig()
        pprint(config)
        return config

    _defaults = {
        key: param.default.default
        for key, param in inspect.signature(predict).parameters.items()
        if hasattr(param.default, "default")
    }
