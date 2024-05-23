import os, random, asyncio, time
from typing import AsyncIterator, List, Union
from cog import BasePredictor, Input, ConcatenateIterator
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
import torch
from utils import maybe_download

from uuid import uuid4
from dotenv import load_dotenv
import yaml

load_dotenv()

PROMPT_TEMPLATE = "<s>[INST] {prompt} [/INST] "

DEFAULT_MAX_NEW_TOKENS = os.getenv("DEFAULT_MAX_NEW_TOKENS", 512)
DEFAULT_TEMPERATURE = os.getenv("DEFAULT_TEMPERATURE", 0.6)
DEFAULT_TOP_P = os.getenv("DEFAULT_TOP_P", 0.9)
DEFAULT_TOP_K = os.getenv("DEFAULT_TOP_K", 50)
DEFAULT_PRESENCE_PENALTY = os.getenv("DEFAULT_PRESENCE_PENALTY", 0.0)  # 1.15
DEFAULT_FREQUENCY_PENALTY = os.getenv("DEFAULT_FREQUENCY_PENALTY", 0.0)  # 0.2


class VLLMPipeline:
    """
    A simplified inference engine that runs inference w/ vLLM
    """

    def __init__(self, *args, **kwargs) -> None:
        args = AsyncEngineArgs(*args, **kwargs)
        self.engine = AsyncLLMEngine.from_engine_args(args)
        self.tokenizer = (
            self.engine.engine.tokenizer.tokenizer
            if hasattr(self.engine.engine.tokenizer, "tokenizer")
            else self.engine.engine.tokenizer
        )

    async def generate_stream(
        self, prompt: str, sampling_params: SamplingParams
    ) -> AsyncIterator[str]:
        results_generator = self.engine.generate(
            prompt, sampling_params, str(random.random())
        )
        async for generated_text in results_generator:
            yield generated_text

    async def __call__(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        stop_sequences: Union[str, List[str]] = None,
        stop_token_ids: List[int] = None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        incremental_generation: bool = True,
    ) -> str:
        """
        Given a prompt, runs generation on the language model with vLLM.
        """
        if top_k is None or top_k == 0:
            top_k = -1

        stop_token_ids = stop_token_ids or []
        stop_token_ids.append(self.tokenizer.eos_token_id)

        if isinstance(stop_sequences, str) and stop_sequences != "":
            stop = [stop_sequences]
        elif isinstance(stop_sequences, list) and len(stop_sequences) > 0:
            stop = stop_sequences
        else:
            stop = []

        for tid in stop_token_ids:
            stop.append(self.tokenizer.decode(tid))

        sampling_params = SamplingParams(
            n=1,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            use_beam_search=False,
            stop=stop,
            max_tokens=max_new_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

        generation_length = 0

        async for request_output in self.generate_stream(prompt, sampling_params):
            assert len(request_output.outputs) == 1
            generated_text = request_output.outputs[0].text
            if incremental_generation:
                yield generated_text[generation_length:]
            else:
                yield generated_text
            generation_length = len(generated_text)


class Predictor(BasePredictor):
    async def setup(self, weights: str = ""):
        
        self.world_size = torch.cuda.device_count()
        self.config = self.init_model_config()
        self.model_path = os.path.join("models", self.config["model_id"])
        self.engine_args = self.get_engine_args()
        
        await self.maybe_download(path=self.model_path, remote_path=self.config["model_url"])

        self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
        self.tokenizer = (
            self.engine.engine.tokenizer.tokenizer
            if hasattr(self.engine.engine.tokenizer, "tokenizer")
            else self.engine.engine.tokenizer
        )


    async def predict(
        self,
        prompt: str = Input(description="Prompt", default=""),
        min_tokens: int = Input(
            description="The minimum number of tokens the model should generate as output.",
            default=DEFAULT_MIN_TOKENS,
        ),
        max_tokens: int = Input(
            description="The maximum number of tokens the model should generate as output.",
            default=DEFAULT_MAX_TOKENS,
        ),
        temperature: float = Input(
            description="The value used to modulate the next token probabilities.",
            default=DEFAULT_TEMPERATURE,
        ),
        top_p: float = Input(
            description="A probability threshold for generating the output. If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751).",
            default=DEFAULT_TOP_P,
        ),
        top_k: int = Input(
            description="The number of highest probability tokens to consider for generating the output. If > 0, only keep the top k tokens with highest probability (top-k filtering).",
            default=DEFAULT_TOP_K,
        ),
        presence_penalty: float = Input(
            description="Presence penalty",
            default=DEFAULT_PRESENCE_PENALTY,
        ),
        frequency_penalty: float = Input(
            description="Frequency penalty",
            default=DEFAULT_FREQUENCY_PENALTY,
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
            stop=stop,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

        generator = self.engine.generate(prompt, sampling_params, request_id)
        async for request_output in generator:
            assert len(request_output.outputs) == 1

            generated_text = request_output.outputs[0].text
            
            # Catches partial emojis, waits for them to finish
            generated_text = generated_text.replace("\N{REPLACEMENT CHARACTER}", "")

            yield generated_text[generation_length:]
            generation_length = len(generated_text)


    def init_model_config(self) -> dict:
        """
        Initializes the model configuration.

        This function loads the model configuration from a YAML file named "config.yaml".
        If the file doesn't exist, it checks for environment variables `MODEL_ID` and `MODEL_URL`.
        If either of these variables is not set, it raises a ValueError.
        The function also sets the `dtype` and `tensor_parallel_size` based on environment variables.
        The default value for `dtype` is "auto" and the default value for `tensor_parallel_size` is `self.world_size`.

        Returns:
            dict: The model configuration dictionary.
        """
        if os.path.exists("config.yaml"):
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
            return config

        else: # Try to build the config from environment variables
            model_id = os.getenv("MODEL_ID", None)
            model_url = os.getenv("COG_WEIGHTS", None)

            if not model_id:
                raise ValueError("No config was provided and `MODEL_ID` is not set.")
            if not model_url:
                raise ValueError("No config was provided and `MODEL_URL` is not set.")
            
            tensor_parallel_size = os.getenv("TENSOR_PARALLEL_SIZE", self.world_size)

            config = {
                "model_id": MODEL_ID,
                "model_url": os.getenv("MODEL_URL", WEIGHTS_URL)
                "dtype": os.getenv("DTYPE", "auto"),
                "tensor_parallel_size": tensor_parallel_size,
            }
        
        return config

    
    async def maybe_download_model(self) -> str:
        if not os.path.exists(f"models/"):
            os.makedirs(f"models/")

        path = self.model_path
        model_url = self.config["model_url"]

        await maybe_download(path=path, remote_path=model_url)
        return path

    def get_engine_args(self) -> AsyncEngineArgs:
        engine_args = self.config.get("engine_args", {})
        engine_args["model"] = self.model_path

        if "dtype" not in engine_args:
            engine_args["dtype"] = "auto"
        
        if "tensor_parallel_size" not in engine_args:
            engine_args["tensor_parallel_size"] = self.world_size

        if "trust_remote_code" not in engine_args:
            engine_args["trust_remote_code"] = True
        
        return AsyncEngineArgs(**engine_args)
        
    def get_sampling_params(self, **kwargs) -> SamplingParams:
        if top_k is None or top_k == 0:
            top_k = -1
        
        stop_token_ids = stop_token_ids or []
        stop_token_ids.append(self.tokenizer.eos_token_id)

        if isinstance(stop_sequences, str) and stop_sequences != "":
            stop = [stop_sequences]
        elif isinstance(stop_sequences, list) and len(stop_sequences) > 0:
            stop = stop_sequences
        else:
            stop = []
        
        for tid in stop_token_ids:
            stop.append(self.tokenizer.decode(tid))

        sampling_params = SamplingParams(
            n=1,
            top_p=kwargs.get("top_p", DEFAULT_TOP_P),
            top_k=kwargs.get("top_k", DEFAULT_TOP_K),
            temperature=kwargs.get("temperature", DEFAULT_TEMPERATURE),
            stop=kwargs.get("stop", []),
            min_tokens=kwargs.get("min_tokens", DEFAULT_MIN_TOKENS),
            max_tokens=kwargs.get("max_tokens", DEFAULT_MAX_TOKENS),
            frequency_penalty=kwargs.get("frequency_penalty", DEFAULT_FREQUENCY_PENALTY),
            presence_penalty=kwargs.get("presence_penalty", DEFAULT_PRESENCE_PENALTY),
            use_beam_search=False,
        )

        return sampling_params


async def main():
    p = Predictor()
    await p.setup()
    async for text in p.predict(
        prompt="Write a blogpost about SEO directed at a technical audience",
        max_new_tokens=512,
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        presence_penalty=1.0,
        frequency_penalty=0.2,
        prompt_template=PROMPT_TEMPLATE,
    ):
        print(text, end="")


if __name__ == "__main__":
    asyncio.run(main())
