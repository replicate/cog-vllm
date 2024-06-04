import io
import json
import tarfile
import time
from collections import namedtuple

import httpx
import tqdm
from cog import BaseModel, Input, Path, Secret
from huggingface_hub import (
    HfApi,
    get_hf_file_metadata,
    hf_hub_url,
)
from huggingface_hub._login import _login as hf_login
from huggingface_hub.utils import filter_repo_objects

from predict import PredictorConfig


class TrainingOutput(BaseModel):
    weights: Path


def train(
    hf_model_id: str = Input(
        description="""
        Hugging Face model identifier 
        (e.g. NousResearch/Hermes-2-Theta-Llama-3-8B).
        """,
    ),
    hf_model_sha: str = Input(
        description="""
        The version of the model.
        If unspecified, the latest version is used.
        """,
        default=None,
    ),
    hf_token: Secret = Input(
        description="""
        Hugging Face API token. 
        Get your token at https://huggingface.co/settings/tokens
        """,
        default=None,
    ),
    allow_patterns: str = Input(
        description="""
        Patterns constituting the allowlist. 
        If provided, item paths must match at least one pattern from the allowlist. 
        (e.g. "*.safetensors").
        """,
        default=None,
    ),
    ignore_patterns: str = Input(
        description="""
        Patterns constituting the denylist. 
        If provided, item paths must not match any patterns from the denylist. 
        (e.g. "*.gguf").
        """,
        default="*.gguf",
    ),
    prompt_template: str = Input(
        description="Prompt template. The string `{prompt}` will be substituted for the input prompt. If you want to generate dialog output, use this template as a starting point and construct the prompt string manually, leaving `prompt_template={prompt}`.",
        default="{prompt}",
    ),
) -> TrainingOutput:
    if hf_token is not None and isinstance(hf_token, Secret):
        print("Logging in to Hugging Face Hub...")
        hf_token = hf_token.get_secret_value().strip()
        hf_login(token=hf_token, add_to_git_credential=False)
    else:
        print("No HuggingFace token provided.")

    api = HfApi()

    # Fetch the model info
    model = api.model_info(
        hf_model_id, revision=hf_model_sha, token=hf_token, files_metadata=True
    )
    print(f"Using model {model.id} with SHA {model.sha}")

    # Determine which files to download
    files = list(
        filter_repo_objects(
            items=[f.rfilename for f in model.siblings],
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
    )
    if len(files) == 0:
        raise ValueError("No files to download")

    Entry = namedtuple("Entry", ["filename", "url", "metadata"])
    entries = [
        Entry(
            filename=x,
            url=hf_hub_url(repo_id=hf_model_id, filename=x),
            metadata=get_hf_file_metadata(
                hf_hub_url(repo_id=hf_model_id, filename=x), token=hf_token
            ),
        )
        for x in files
    ]

    config = PredictorConfig(prompt_template=prompt_template)

    start = time.time()
    print(f"Downloading {len(files)} files...")

    # Download the files and write them to a tar file
    weights = Path("model.tar")
    with tarfile.open(name=str(weights), mode="w:") as tar:
        # Add predictor_config.json
        predictor_config_data = json.dumps(config._asdict()).encode("utf-8")
        tar_info = tarfile.TarInfo("predictor_config.json")
        tar_info.mtime = int(time.time())
        tar_info.size = len(predictor_config_data)
        tar.addfile(
            tar_info=tar_info,
            fileobj=io.BytesIO(predictor_config_data),
        )

        with tqdm.tqdm(
            total=sum(entry.metadata.size for entry in entries),
            unit="B",
            unit_divisor=1024,
            unit_scale=True,
            mininterval=1,
        ) as pbar:
            headers = {"Authorization": f"Bearer {hf_token}"}
            with httpx.Client(
                headers=headers, follow_redirects=True, timeout=None
            ) as client:
                for n, entry in enumerate(entries, start=1):
                    pbar.update(0)
                    pbar.set_postfix(
                        n=f"{n}/{len(entries)}",
                        file=entry.filename,
                        refresh=True,
                    )

                    with client.stream("GET", entry.url) as response:
                        response.raise_for_status()

                        with io.BytesIO() as buffer:
                            for chunk in response.iter_bytes(chunk_size=32 * 1024):
                                buffer.write(chunk)

                                pbar.update(len(chunk))
                                pbar.set_postfix(
                                    n=f"{n}/{len(entries)}",
                                    file=entry.filename,
                                    refresh=False,
                                )

                            buffer.seek(0)

                            tar_info = tarfile.TarInfo(entry.filename)
                            tar_info.mtime = int(time.time())
                            tar_info.size = entry.metadata.size
                            tar.addfile(tar_info, fileobj=buffer)

        print(f"Downloaded {len(files)} files in {time.time() - start:.2f} seconds")

    return TrainingOutput(weights=weights)
