import asyncio
import os
import typing as tp
import subprocess
import time

def check_files_exist(remote_files: list[str], local_path: str) -> list[str]:
    local_files = os.listdir(local_path)
    missing_files = list(set(remote_files) - set(local_files))
    return missing_files


async def download_files_with_pget(
    remote_path: str, path: str, files: list[str]
) -> None:

    # get all the files that are not .tars
    download_jobs = "\n".join(
        f"{remote_path}/{f} {path}/{f}" for f in files if not f.endswith(".tar")
    )
    args = ["pget", "multifile", "-", "-f", "--max-conn-per-host", "100"]
    start_time = time.time()
    process = await asyncio.create_subprocess_exec(*args, stdin=-1, close_fds=True)
    # Wait for the subprocess to finish
    await process.communicate(download_jobs.encode())
    print(f"Downloaded model assets in {time.time() - start_time:.2f}s")

    return path


async def maybe_download_tarball_with_pget(
    tarball_url: str,
    destination_path: str,
):
    """
    Downloads a tarball from `tarball_url` and extracts to `destination_path` if `destination_path` does not exist. If remote_path is None, files are not downloaded.

    Args:
        tarball_url (str): URL to the tarball
        destination_path (str): Path to the directory where the tarball should be decompressed

    Returns:
        path (str): Path to the directory where files were downloaded

    """

    # if destination_path exists and is not empty, return
    if os.path.exists(destination_path) and os.listdir(destination_path):
        print(f"Files already present in the `{destination_path}`, nothing will be downloaded.")
        return destination_path

    # if destination_path exists but is empty, remove it so we can pull with pget
    if os.path.exists(destination_path):
        shutil.rmtree(destination_path)

    print("Downloading model assets...")
    start_time = time.time()
    command = ["pget", tarball_url, destination_path, "-x"]
    subprocess.check_call(command, close_fds=True)
    print(f"Downloaded model assets in {time.time() - start_time:.2f}s")

    return destination_path


async def maybe_download(
    path: str,
    remote_path: tp.Optional[str] = None,
    model_id: tp.Optional[str] = None,
    remote_filenames: tp.Optional[list[str]] = None,
):
    """
    Downloads files from a remote location if necessary.
    If `remote_path` is a tarball, it is downloaded and extracted. If not, the files in `remote_filenames` are downloaded.

    Args:
        path (str): The local path where the files will be downloaded.
        remote_path (str, optional): The remote path where the files are located. Defaults to None.
        remote_filenames (list[str], optional): The list of filenames to be downloaded. Required if `remote_path` is not a tarball. Defaults to None.

    Returns:
        str: The local path where the files are downloaded.

    """

    if remote_path:
        remote_path = remote_path.rstrip("/")

        if remote_path.endswith(".tar"):
            path = await maybe_download_and_extract_tarball(remote_path, path)
            return path

        else:
            assert remote_filenames is not None, "remote_filenames must be provided if `remote_path` is not a tarball"
            
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                missing_files = remote_filenames
            
            else:
                missing_files = check_files_exist(remote_filenames, path)
            
            path = await maybe_download_files(remote_path, path, missing_files)
            return path
            
    else:
        raise NotImplementedError("Downloading from HF not implemented yet, please provide a remote path to a flat tarball containing your model artifacts.")
