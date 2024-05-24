import os, random, asyncio, time
import typing as tp
from cog import BasePredictor, Input, ConcatenateIterator
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
import torch

from uuid import uuid4
from dotenv import load_dotenv
import yaml

from utils import maybe_download, init_model_config
from prompt_templates import get_prompt_template

config = init_model_config()

MODEL_ID = config["model_id"].split("/")[-1]
DEFAULT_PROMPT_TEMPLATE = config.get("prompt_template", None)
if not DEFAULT_PROMPT_TEMPLATE or DEFAULT_PROMPT_TEMPLATE == "":
    DEFAULT_PROMPT_TEMPLATE = get_prompt_template(MODEL_ID)

IS_INSTRUCT = config.get("is_instruct", True)
DEFAULT_SYSTEM_PROMPT = config.get("system_prompt", None)
if IS_INSTRUCT and not DEFAULT_SYSTEM_PROMPT:
    DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."




class Predictor(BasePredictor):
    async def setup(self):
        
        self.world_size = torch.cuda.device_count()
        self.config = init_model_config()
        self.model_path = os.path.join("models", self.config["model_id"])
        self.engine_args = self.get_engine_args()
        
        await self.maybe_download_model()

        self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
        self.tokenizer = (
            self.engine.engine.tokenizer.tokenizer
            if hasattr(self.engine.engine.tokenizer, "tokenizer")
            else self.engine.engine.tokenizer
        )


    async def predict(
        self,
        prompt: str = Input(description="Prompt", default=""),
        system_prompt: str = Input(
            description="System prompt to send to the model. This is prepended to the prompt and helps guide system behavior.",
            default=DEFAULT_SYSTEM_PROMPT,
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
        presence_penalty: float = Input(
            description="Presence penalty",
            default=0.0,
        ),
        frequency_penalty: float = Input(
            description="Frequency penalty",
            default=0.0,
        ),
        stop_sequences: str = Input(
            description="A comma-separated list of sequences to stop generation at. For example, '<end>,<stop>' will stop generation at the first instance of 'end' or '<stop>'.",
            default=None,
        ),
        prompt_template: str = Input(
            description="Prompt template. The string `{prompt}` will be substituted for the input prompt. If you want to generate dialog output, use this template as a starting point and construct the prompt string manually, leaving `prompt_template={prompt}`.",
            default=DEFAULT_PROMPT_TEMPLATE,
        ),
    ) -> ConcatenateIterator[str]:
        start = time.time()
        request_id = uuid4().hex
        generation_length = 0

        sampling_params = self.get_sampling_params(
            n=1,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            use_beam_search=False,
            stop_sequences=stop_sequences,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

        if self.config["is_instruct"]:
            prompt = prompt_template.format(prompt=prompt, system_prompt=system_prompt)
        else:
            prompt = prompt_template.format(prompt=prompt)


        generator = self.engine.generate(prompt, sampling_params, request_id)
        async for request_output in generator:
            assert len(request_output.outputs) == 1

            generated_text = request_output.outputs[0].text
            
            # Catches partial emojis, waits for them to finish
            generated_text = generated_text.replace("\N{REPLACEMENT CHARACTER}", "")

            yield generated_text[generation_length:]
            generation_length = len(generated_text)
        
        print(f"Generation took {time.time() - start:.2f}s")
        print(f"Formatted prompt: {prompt}")

    
    async def maybe_download_model(self) -> str:
        if not os.path.exists(f"models/"):
            os.makedirs(f"models/")

        path = self.model_path
        model_url = self.config["model_url"]

        await maybe_download(path=path, remote_path=model_url)
        return path

    def get_engine_args(self) -> AsyncEngineArgs:
        engine_args = self.config.get("engine_args", {})
        if not engine_args:
            engine_args = {}
            
        engine_args["model"] = self.model_path

        if "dtype" not in engine_args:
            engine_args["dtype"] = "auto"
        
        if "tensor_parallel_size" not in engine_args:
            engine_args["tensor_parallel_size"] = self.world_size

        if "trust_remote_code" not in engine_args:
            engine_args["trust_remote_code"] = True
        
        return AsyncEngineArgs(**engine_args)
        
    def get_sampling_params(self, **kwargs) -> SamplingParams:

        top_k = kwargs.get("top_k", None)
        if top_k is None or top_k == 0:
            top_k = -1
        
        stop_token_ids = kwargs.get("stop_token_ids", None)
        stop_token_ids = stop_token_ids or []
        stop_token_ids.append(self.tokenizer.eos_token_id)
        
        stop_sequences = kwargs.get("stop_sequences", None)
        if isinstance(stop_sequences, str) and stop_sequences != "":
            stop = stop_sequences.split(",")
        elif isinstance(stop_sequences, list) and len(stop_sequences) > 0:
            stop = stop_sequences
        else:
            stop = []
        
        sampling_params = SamplingParams(
            n=1,
            stop=stop,
            stop_token_ids=stop_token_ids,
            top_k=top_k,
            top_p=kwargs.get("top_p"),
            temperature=kwargs.get("temperature"),
            min_tokens=kwargs.get("min_tokens"),
            max_tokens=kwargs.get("max_tokens"),
            frequency_penalty=kwargs.get("frequency_penalty"),
            presence_penalty=kwargs.get("presence_penalty"),
            use_beam_search=False,
        )

        return sampling_params

    def remove(f: tp.Callable, defaults: tp.Dict[str, tp.Any]) -> tp.Callable:
        # pylint: disable=no-self-argument
        def wrapper(self, *args, **kwargs):
            kwargs.update(defaults)
            return f(self, *args, **kwargs)

        # Update wrapper attributes for documentation, etc.
        functools.update_wrapper(wrapper, f)

        # for the purposes of inspect.signature as used by predictor.get_input_type,
        # remove the argument (system_prompt)
        sig = inspect.signature(f)
        params = [p for name, p in sig.parameters.items() if name not in defaults]
        wrapper.__signature__ = sig.replace(parameters=params)

        # Return partialmethod, wrapper behaves correctly when part of a class
        return functools.partialmethod(wrapper)

    args_to_remove: dict[str, tp.Any] = {}
    if not IS_INSTRUCT:
        # this removes system_prompt from the Replicate API for non-chat models.
        args_to_remove["system_prompt"] = None
    if args_to_remove:
        predict = remove(predict, args_to_remove)
