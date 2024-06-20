import os
import subprocess
import time
import warnings
from urllib.parse import urlparse


async def resolve_model_path(url_or_local_path: str) -> str:
    """
    Resolves the model path, downloading if necessary.

    Args:
        url_or_local_path (str): URL to the tarball or local path to a directory containing the model artifacts.

    Returns:
        str: Path to the directory containing the model artifacts.
    """

    parsed_url = urlparse(url_or_local_path)
    if parsed_url.scheme == "http" or parsed_url.scheme == "https":
        return await download_tarball(url_or_local_path)
    elif parsed_url.scheme == "file" or parsed_url.scheme == "":
        if not os.path.exists(parsed_url.path):
            raise ValueError(
                f"E1000: The provided local path '{parsed_url.path}' does not exist."
            )
        if not os.listdir(parsed_url.path):
            raise ValueError(
                f"E1000: The provided local path '{parsed_url.path}' is empty."
            )

        warnings.warn(
            "Using local model artifacts for development is okay, but not optimal for production. "
            "To minimize boot time, store model assets externally on Replicate."
        )
        return url_or_local_path
    else:
        raise ValueError(f"E1000: Unsupported model path scheme: {parsed_url.scheme}")


async def download_tarball(url: str) -> str:
    """
    Downloads a tarball from a URL and extracts it.

    Args:
        url (str): URL to the tarball.

    Returns:
        str: Path to the directory where the tarball was extracted.
    """
    filename = os.path.splitext(os.path.basename(url))[0]
    path = os.path.join(os.getcwd(), "models", filename)
    if os.path.exists(path) and os.listdir(path):
        print(f"Files already present in `{path}`.")
        return path

    print(f"Downloading model assets to {path}...")
    start_time = time.time()
    command = ["pget", url, path, "-x"]
    subprocess.check_call(command, close_fds=True)
    print(f"Downloaded model assets in {time.time() - start_time:.2f}s")
    return path
