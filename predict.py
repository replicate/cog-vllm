import os, random, asyncio, time
from typing import AsyncIterator, List, Union
from cog import BasePredictor, Input, ConcatenateIterator
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
import torch
from utils import maybe_download_with_pget

from dotenv import load_dotenv

load_dotenv()

MODEL_ID = os.getenv("MODEL_ID")
WEIGHTS_URL = os.getenv("COG_WEIGHTS")
REMOTE_FILES = [
    "model.tar",
]
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
    async def setup(self):
        n_gpus = torch.cuda.device_count()
        start = time.time()
        await maybe_download_with_pget(MODEL_ID, WEIGHTS_URL, REMOTE_FILES)
        print(f"downloading weights took {time.time() - start:.3f}s")
        self.llm = VLLMPipeline(
            tensor_parallel_size=n_gpus,
            model=MODEL_ID,
            # quantization="awq",
            dtype="auto",
            trust_remote_code=True,
            # max_model_len=MAX_TOKENS
        )

    async def predict(
        self,
        prompt: str,
        max_new_tokens: int = Input(
            description="The maximum number of tokens the model should generate as output.",
            default=DEFAULT_MAX_NEW_TOKENS,
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
            description="The template used to format the prompt. The input prompt is inserted into the template using the `{prompt}` placeholder.",
            default=PROMPT_TEMPLATE,
        ),
    ) -> ConcatenateIterator[str]:
        start = time.time()
        generate = self.llm(
            prompt=prompt_template.format(prompt=prompt),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        async for text in generate:
            yield text
        print(f"generation took {time.time() - start:.3f}s")


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
