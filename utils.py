import asyncio
import os
import typing as tp
import subprocess
import shutil
import time
import torch
import yaml
from typing import Callable


async def download_and_extract_tarball(
    url: str,
) -> str:
    """
    Downloads a tarball from `url` and extracts to `dirname` if `dirname` does not exist.

    Args:
        url (str): URL to the tarball

    Returns:
        path (str): Path to the directory where the tarball was extracted
    """

    # if path exists and is not empty, return
    if os.path.exists(url) and os.listdir(url):
        print(f"Files already present in the `{url}`, nothing will be downloaded.")
        return url

    path = os.path.basename(url)

    if os.path.exists(path):
        raise ValueError(f"Path {path} already exists")

    print(f"Downloading model assets to {path}...")
    start_time = time.time()
    command = ["pget", url, path, "-x"]
    subprocess.check_call(command, close_fds=True)
    print(f"Downloaded model assets in {time.time() - start_time:.2f}s")

    return path


def remove_system_prompt_input(f: Callable) -> Callable:
    import functools
    import inspect

    def wrapper(self, *args, **kwargs):
        if "system_prompt" in kwargs:
            del kwargs["system_prompt"]
        return f(self, *args, **kwargs)

    functools.update_wrapper(wrapper, f)

    sig = inspect.signature(f)
    params = [p for name, p in sig.parameters.items() if name != "system_prompt"]
    wrapper.__signature__ = sig.replace(parameters=params)

    return functools.partialmethod(wrapper)
