# pylint: disable=duplicate-code
import os
import subprocess
import time
import warnings
from urllib.parse import urlparse
from pathlib import Path
import asyncio
import shutil


async def resolve_model_path(url_or_local_path: str) -> str:
    """
    Resolves the model path, downloading if necessary.

    Args:
        url_or_local_path (str): URL to the tarball or local path to a
        directory containing the model artifacts.

    Returns:
        str: Path to the directory containing the model artifacts.
    """

    parsed_url = urlparse(url_or_local_path)
    if parsed_url.scheme in ["http", "https"]:
        return await download_tarball(url_or_local_path)

    if parsed_url.scheme in ["file", ""]:
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
    raise ValueError(f"E1000: Unsupported model path scheme: {parsed_url.scheme}")


async def maybe_download_tarball_with_pget(
    url: str,
    dest: str,
):
    """
    Checks for existing model weights in a local volume or checkpoints cache,
    downloads if necessary, and sets up symlinks.

    This function first checks if weights exist in the local checkpoints cache.
    If not, it checks for a local volume (/weights) and uses that if available.
    If the weights already exist in any location, no download occurs.

    Args:
        url (str): URL to the model tarball.
        dest (str): Destination path for the weights (e.g., ./checkpoints/{model_name}).

    Returns:
        str: Path to the directory containing the model weights, which may be either
             the original destination or a symlink to the local volume.

    Note:
        - Prioritizes using existing cache in ./checkpoints/ over /weights volume
        - If weights are in the local volume, a symlink is created to `dest`.
        - If weights are already present in either location, no download occurs.
    """
    # First check if dest (checkpoints cache) already exists and has files
    if os.path.exists(dest) and os.listdir(dest):
        print(f"Files already present in `{dest}`, using cached version.")
        return dest

    try:
        Path("/weights").mkdir(exist_ok=True)
        first_dest = "/weights/vllm"
    except PermissionError:
        print("/weights doesn't exist, and we couldn't create it")
        first_dest = dest

    # if first_dest (/weights/vllm) exists and is not empty, use it
    if os.path.exists(first_dest) and os.listdir(first_dest):
        print(f"Files already present in `{first_dest}`, nothing will be downloaded.")
        if first_dest != dest:
            try:
                if os.path.islink(dest):
                    os.unlink(dest)
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
    # subprocess.check_call(command, close_fds=True)

    process = await asyncio.create_subprocess_exec(*command, close_fds=True)
    await process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)

    print(f"Downloaded model assets in {time.time() - start_time:.2f}s")
    if first_dest != dest:
        if os.path.islink(dest):
            os.unlink(dest)
        os.symlink(first_dest, dest)

    return dest


async def download_tarball(url: str) -> str:
    """
    Downloads a tarball from a URL and extracts it to ./checkpoints/{model_name}.

    This implements local caching - models are stored in the checkpoints directory
    and won't be re-downloaded if they already exist.

    Args:
        url (str): URL to the tarball.

    Returns:
        str: Path to the directory where the tarball was extracted.
    """
    # Extract model name from URL
    # Example: https://weights.replicate.delivery/default/Qwen/Qwen3-VL-8B-Instruct/model.tar
    # Should extract: Qwen3-VL-8B-Instruct
    url_parts = urlparse(url)
    path_parts = url_parts.path.strip('/').split('/')

    # Try to find model name in URL path
    model_name = None
    if 'model.tar' in path_parts[-1]:
        # Model name is likely the directory before model.tar
        if len(path_parts) >= 2:
            model_name = path_parts[-2]
    else:
        # Use the last part of the URL (without .tar extension)
        filename = os.path.splitext(os.path.basename(url))[0]
        model_name = filename

    # Create checkpoints directory if it doesn't exist
    checkpoints_dir = os.path.join(os.getcwd(), 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Set destination to checkpoints/{model_name}
    dest = os.path.join(checkpoints_dir, model_name)

    return await maybe_download_tarball_with_pget(url, dest)
