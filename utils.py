import os
import subprocess
import time
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

    filename = os.path.splitext(os.path.basename(url))[0]
    path = os.path.join(os.getcwd(), "models", filename)

    if os.path.exists(path) and os.listdir(path):
        print(f"Files already present in the `{path}`, nothing will be downloaded.")
        return path

    print(f"Downloading model assets to {path}...")
    start_time = time.time()
    command = ["pget", url, path, "-x"]
    subprocess.check_call(command, close_fds=True)
    print(f"Downloaded model assets in {time.time() - start_time:.2f}s")

    return path


def remove_system_prompt_input(f: Callable) -> Callable:
    import functools
    import inspect

    # we could use inspect.is{asyncgen,coroutine,generator}function
    # but this is just for vLLM which should always be async def -> AsyncIterator
    async def wrapper(self, *args, **kwargs):
        if "system_prompt" in kwargs:
            del kwargs["system_prompt"]
        async for item in f(self, *args, **kwargs):
            yield item

    functools.update_wrapper(wrapper, f)

    sig = inspect.signature(f)
    params = [p for name, p in sig.parameters.items() if name != "system_prompt"]
    wrapper.__signature__ = sig.replace(parameters=params)

    return functools.partialmethod(wrapper)
