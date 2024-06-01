import io
import os
import tarfile
import time
from collections import namedtuple
from typing import List, Union

import requests
import tqdm
from cog import BaseModel, Input, Path, Secret

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HOME"] = "./hf-cache"

from huggingface_hub import (
    HfApi,
    get_hf_file_metadata,
    hf_hub_url,
)
from huggingface_hub._login import _login as hf_login
from huggingface_hub.utils import filter_repo_objects


class TrainingOutput(BaseModel):
    weights: Path


def train(
    hf_model_id: str = Input(
        description="""
        Hugging Face model identifier 
        (e.g. NousResearch/Hermes-2-Theta-Llama-3-8B)
        """,
        regex=r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$",
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
        Get your token at https://huggingface.co/settings/tokens.
        """,
        default=None,
    ),
    allow_patterns: str = Input(
        description="""
        Patterns constituting the allowlist. 
        If provided, item paths must match at least one pattern from the allowlist. 
        (e.g. "*.bin,*.safetensors").
        """,
        default=None,
    ),
    ignore_patterns: str = Input(
        description="""
        Patterns constituting the denylist. 
        If provided, item paths must not match any patterns from the denylist. 
        (e.g. "*.pdf").
        """,
        default=None,
    ),
) -> TrainingOutput:
    if hf_token is not None and isinstance(hf_token, Secret):
        print("Logging in to Hugging Face Hub...")
        hf_token = hf_token.get_secret_value().strip()
        hf_login(token=hf_token, add_to_git_credential=False)
    else:
        print(
            "No HuggingFace token provided. "
            "Private tokenizers and models won't be accessible."
        )

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
    else:
        print(f"Downloading {len(files)} files")

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

    # Download the files and write them to a tar file
    weights = Path("model.tar")
    with tarfile.open(name=str(weights), mode="w:") as tar:
        with tqdm.tqdm(
            total=sum(entry.metadata.size for entry in entries),
            unit="B",
            unit_divisor=1024,
            unit_scale=True,
            mininterval=1,
        ) as pbar:
            for i, entry in enumerate(entries):
                with requests.get(entry.url, stream=True, timeout=None) as response:
                    response.raise_for_status()

                    with io.BytesIO() as buffer:
                        for chunk in response.iter_content(chunk_size=32 * 1024):
                            buffer.write(chunk)
                            pbar.set_postfix(
                                n=f"{i}/{len(entries)}",
                                file=entry.filename[-20:],
                                refresh=False,
                            )
                            pbar.update(len(chunk))

                        buffer.seek(0)

                        tar_info = tarfile.TarInfo(entry.filename)
                        tar_info.mtime = int(time.time())
                        tar_info.size = entry.metadata.size
                        tar.addfile(tar_info, fileobj=buffer)

    return TrainingOutput(weights=weights)