import os
import subprocess
import time
import warnings


async def download_and_extract_tarball(
    url_or_local_path: str,
) -> str:
    """
    Downloads a tarball from `url` and extracts to `dirname` if `dirname` does not exist.

    Args:
        url_or_local_path (str): URL to the tarball or local path to a directory containing the model artifacts

    Returns:
        path (str): If `url_or_local_path` is a local path, returns the path. Otherwise, returns the path to the directory where the tarball was extracted.
    """
    if "://" in url_or_local_path:
        path_is_local = False
        filename = os.path.splitext(os.path.basename(url_or_local_path))[0]
        path = os.path.join(os.getcwd(), "models", filename)
    else:
        path_is_local = True
        path = url_or_local_path

    if path:
        warnings.warn(
            f"The provided path appears to be local. For local development, this is acceptable. However, for production, we strongly advise storing model assets externally to improve boot times.",
            UserWarning,
        )
    if os.path.exists(path) and os.listdir(path):
        print(f"Files already present in the `{path}`.")
        return path
    elif path_is_local and not os.path.exists(path):
        raise ValueError(
            f"E1000 GenericError: The provided local path '{path}' does not exist. Please provide a local path with the model artifacts or a URL to a tarball containing the model artifacts."
        )
    elif os.path.exists(path) and path_is_local:
        raise ValueError(
            f"E1000 GenericError: The provided local path '{path}' is empty. Please provide a local path with the model artifacts or a URL to a tarball containing the model artifacts."
        )

    print(f"Downloading model assets to {path}...")
    start_time = time.time()
    command = ["pget", url_or_local_path, path, "-x"]
    subprocess.check_call(command, close_fds=True)
    print(f"Downloaded model assets in {time.time() - start_time:.2f}s")

    return path
