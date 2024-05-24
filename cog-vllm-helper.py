#!/usr/bin/env python3
import argparse
import yaml
import subprocess
import platform
import shutil
import typing as tp
import pprint

def str_presenter(dumper, data):
    if data.count('\n') > 0:  # Check for multiline string
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

yaml.add_representer(str, str_presenter) 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        default=None,
        help="Model ID to use (e.g., mistralai/Mistral-7B-Instruct-v0.2). Hugging Face compatible, but can be an arbitrary string.",
    )

    parser.add_argument(
        "--model-url",
        type=str,
        required=True,
        default=None,
        help="A URL pointing to a flat tarball containing all model artifacts (e.g. weights, tokenizer, configs, etc).",
    )

    parser.add_argument(
        "--is-instruct",
        action="store_true",
        required=False,
        default=None,
        help="If the model you're deploying uses a system-prompt, set this to true. If you do not set this, cog-vllm will attempt to infer it from the model ID.",
    )

    parser.add_argument(
        "--prompt-template",
        type=bool,
        required=False,
        default=None,
        help="The default prompt template to use for the model. If you do not set this, cog-vllm will attempt to infer it from the model ID.",
    )

    parser.add_argument(
        "--system-prompt",
        type=str,
        required=False,
        default=None,
        help='If the model you\'re deploying uses a system-prompt, set this to the desired system prompt. If you do not set this and your model uses a system prompt, `"You are a helpful assistant."` will be used.',
    )

    parser.add_argument(
        "-t",
        "--tag-name",
        type=str,
        required=False,
        default="cog-vllm",
        help="Tag name for the COG build.",
    )

    parser.add_argument(
        "--",
        dest="engine_args",
        nargs=argparse.REMAINDER,
        help="Arbitrary keyword arguments for the engine in the format --key value.",
    )

    # Parse known and unknown arguments
    args, unknown = parser.parse_known_args()

    if args.model_id is None and args.model_url is None:
        raise ValueError("Either model ID or model URL must be specified.")

    if not args.model_id:
        args.model_id = "model"

    if not args.model_url:
        raise NotImplementedError(
            "Mirroring model weights from Hugging Face is not yet supported. Please provide a model URL."
        )

    # Convert unknown args into a dictionary
    engine_args = {}
    it = iter(unknown)
    for key in it:
        if key.startswith("--"):
            key = key.lstrip("--").replace("-", "_")
            value = next(it, None)
            # Infer type
            if value is not None:
                if value.isdigit():
                    value = int(value)
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            engine_args[key] = value

    args.engine_args = engine_args
    return args


def install_cog_if_needed():
    if shutil.which("cog") is None:
        cog_url = f"https://github.com/replicate/cog/releases/download/v0.10.0-alpha5/cog_{platform.system().lower()}_{platform.machine().lower()}"
        subprocess.run(
            ["sudo", "curl", "-o", "/usr/local/bin/cog", "-L", cog_url], check=True
        )
        subprocess.run(["sudo", "chmod", "+x", "/usr/local/bin/cog"], check=True)
    # else:
        print("Cog is already installed.")


def main(
    model_id: tp.Optional[str] = None,
    model_url: tp.Optional[str] = None,
    is_instruct: tp.Optional[bool] = None,
    prompt_template: tp.Optional[str] = None,
    system_prompt: tp.Optional[str] = None,
    engine_args: tp.Optional[tp.Dict[str, tp.Any]] = None,
    tag_name: str = "cog-vllm",
):
    output = "config.yaml"

    config = {
        "model_id": model_id,
        "model_url": model_url,
        "is_instruct": is_instruct,
        "prompt_template": prompt_template,
        "system_prompt": system_prompt,
    }

    if engine_args:
        config["engine_args"] = engine_args

    # Remove keys with None value from config
    config = {k: v for k, v in config.items() if v is not None}

    with open(output, "w") as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False, sort_keys=False, width=float("inf"))

    install_cog_if_needed()

    # Build with Cog using the specified tag name
    subprocess.run(["cog", "build", "-t", tag_name], check=True)

    print("------" * 10)
    print("Generated config.yaml:")
    print("-------------------------")
    pprint.pprint(config)
    print("------" * 10)


if __name__ == "__main__":

    args = parse_args()

    main(
        model_id=args.model_id,
        model_url=args.model_url,
        is_instruct=args.is_instruct,
        prompt_template=args.prompt_template,
        system_prompt=args.system_prompt,
        engine_args=args.engine_args,
        tag_name=args.tag_name,
    )
