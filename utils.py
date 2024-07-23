import os
import shutil
import subprocess
import time
import warnings
from pathlib import Path
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

def maybe_download_tarball_with_pget(
    url: str,
    dest: str,
):
    """
    Downloads a tarball from url and decompresses to dest if dest does not exist. Remote path is constructed
    by concatenating remote_path and remote_filename. If remote_path is None, files are not downloaded.

    Args:
        url (str): URL to the tarball
        dest (str): Path to the directory where the tarball should be decompressed

    Returns:
        path (str): Path to the directory where files were downloaded

    """
    try:
        Path("/weights").mkdir(exist_ok=True)
        first_dest = "/weights/vllm"
    except PermissionError:
        print("/weights doesn't exist, and we couldn't create it")
        first_dest = dest


    # if dest exists and is not empty, return
    if os.path.exists(first_dest) and os.listdir(first_dest):
        print(f"Files already present in `{first_dest}`, nothing will be downloaded.")
        if first_dest != dest:
            try:
                os.symlink(first_dest, dest)
            except FileExistsError:
                print(f"Ignoring existing file at {dest}")
        return dest

    # if dest exists but is empty, remove it so we can pull with pget
    if os.path.exists(first_dest):
        shutil.rmtree(first_dest)

    print("Downloading model assets...")
    start_time = time.time()
    command = ["pget", url, first_dest, "-x"]
    subprocess.check_call(command, close_fds=True)
    print(f"Downloaded model assets in {time.time() - start_time:.2f}s")
    if first_dest != dest:
        os.symlink(first_dest, dest)

    return dest


async def download_tarball(url: str) -> str:
    """
    Downloads a tarball from a URL and extracts it.

    Args:
        url (str): URL to the tarball.

    Returns:
        str: Path to the directory where the tarball was extracted.
    """
    filename = os.path.splitext(os.path.basename(url))[0]
    dest = os.path.join(os.getcwd(), "models", filename)
    return maybe_download_tarball_with_pget(url, dest)

