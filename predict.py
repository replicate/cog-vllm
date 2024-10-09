import json
import os
import time
from typing import NamedTuple, Optional
from openai import AsyncOpenAI
import httpx

# import torch
from cog import BasePredictor, ConcatenateIterator, Input

from utils import resolve_model_path


SYSTEM_PROMPT = "You are a helpful assistant."

TRITON_START_TIMEOUT_MINUTES = 5
TRITON_TIMEOUT = 120

class PredictorConfig(NamedTuple):
    prompt_template: Optional[str] = None


class Predictor(BasePredictor):
    async def start_vllm(self, model: str, args: dict) -> bool:
        client = httpx.Client()
        cmd = [f"{os.environ['PYTHON_VLLM']}/bin/vllm", "serve", model]
        for k, v in args.items():
            cmd.extend(("--" + k, v))
        self.proc = subprocess.Popen(cmd)
        # Health check Triton until it is ready or for 5 minutes
        for i in range(TRITON_START_TIMEOUT_MINUTES * 60):
            try:
                response = await client.get(
                    "http://localhost:8000/v1/completions"
                )
                if response.status_code == 200:
                    print("VLLM is ready.")
                    return True
            except httpx.RequestError:
                pass
            await asyncio.sleep(1)
        print(f"VLLM was not ready within {TRITON_START_TIMEOUT_MINUTES} minutes (exit code: {self.proc.poll()})")
        self.proc.terminate()
        await asyncio.sleep(0.001)
        self.proc.kill()
        return False
    
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
        res = self.start_vllm(
            weights,
            {
                "dtype": "auto",
                "tensor_parallel_size": max(torch.cuda.device_count(), 1),
                "max_model_len": 4096,
            }
        )

        
        self.client = AsyncOpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1"
        )
        models = self.client.models.list()
        self.model_id = models.data[0].id

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
    ) -> ConcatenateIterator[str]:
        start = time.time()

        system_prompt = "" if system_prompt is None else system_prompt
        if isinstance(stop_sequences, str) and stop_sequences:
            stop = stop_sequences.split(",")
        else:
            stop = (
                list(stop_sequences) if isinstance(stop_sequences, list) else []
            )
        chat_response = await self.client.chat.completions.create(
            model=self.model_id,
            echo=False,
            n=1,
            stream=True,
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            top_k=(-1 if (top_k or 0) == 0 else top_k),
            top_p=top_p,
            stop=stop,
            temperature=temperature,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            use_beam_search=False,
        )
        async for completion in stream:
            yield completion.choices[0].text
            
        self.log(f"Generation took {time.time() - start:.2f}s")
        self.log(f"Formatted prompt: {prompt}")
