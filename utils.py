import os
import subprocess
import time


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
