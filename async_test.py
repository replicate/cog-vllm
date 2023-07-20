from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams

import argparse

import json


class Engine:
    def __init__(self, engine):
        self.engine = engine

    def generate_stream(
        self,
        prompt,
        request_id,
        temperature=1.0,
        top_p=1.0,
        max_new_tokens=256,
        stop_str=None,
        stop_token_ids=None,
        echo=True,
    ):
        self.call_ct += 1

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
        )
        results_generator = engine.generate(context, sampling_params, request_id)

        for request_output in results_generator:
            prompt = request_output.prompt
            if echo:
                text_outputs = [
                    prompt + output.text for output in request_output.outputs
                ]
            else:
                text_outputs = [output.text for output in request_output.outputs]
            text_outputs = " ".join(text_outputs)
            # Note: usage is not supported yet
            ret = {"text": text_outputs, "error_code": 0, "usage": {}}
            yield (json.dumps(ret) + "\0").encode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path", type=str, default="meta-llama/Llama-2-7b-chat-hf"
    )
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )

    parser.add_argument("--num-gpus", type=int, default=1)

    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    if args.model_path:
        args.model = args.model_path
    if args.num_gpus > 1:
        args.tensor_parallel_size = args.num_gpus

    engine_args = AsyncEngineArgs.from_cli_args(args)
    print(f"Engine args: {engine_args}")

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print(f"Made einge")

    engine = Engine(engine)
    for v in engine.generate_stream("User: Hello\nAssistant: ", 1):
        print(v)
