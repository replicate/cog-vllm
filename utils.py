import asyncio
import os
import typing as tp
import subprocess
import time
import torch
import yaml

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


async def maybe_download_and_extract_tarball(
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


def init_model_config() -> dict:
        """
        Initializes the model configuration.

        This function loads the model configuration from a YAML file named "config.yaml".
        If the file doesn't exist, it checks for environment variables `MODEL_ID` and `MODEL_URL`.
        If either of these variables is not set, it raises a ValueError.
        The function also sets the `dtype` and `tensor_parallel_size` based on environment variables.
        The default value for `dtype` is "auto" and the default value for `tensor_parallel_size` is `self.world_size`.

        Returns:
            dict: The model configuration dictionary.
        """
        if os.path.exists("config.yaml"):
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)

        else: # Try to build the config from environment variables
            model_id = os.getenv("MODEL_ID", None)
            model_url = os.getenv("COG_WEIGHTS", None)
            is_instruct = os.getenv("IS_INSTRUCT", None)

            if not model_id:
                raise ValueError("No config was provided and `MODEL_ID` is not set.")
            if not model_url:
                raise ValueError("No config was provided and `MODEL_URL` is not set.")
            
            
            world_size = torch.cuda.device_count()
            tensor_parallel_size = os.getenv("TENSOR_PARALLEL_SIZE", world_size)

            config = {
                "model_id": MODEL_ID,
                "model_url": os.getenv("MODEL_URL", WEIGHTS_URL),
                "dtype": os.getenv("DTYPE", "auto"),
                "tensor_parallel_size": tensor_parallel_size,
            }

            if is_instruct:
                config["is_instruct"] = is_instruct
        
        if "is_instruct" not in config:
            print("`is_instruct` not specified, attempting to infer from `model_id` and `model_url`.")
            config["is_instruct"] = any(
                keyword in config[key] 
                for keyword in ["chat", "instruct"] 
                for key in ["model_id", "model_url"]
            )

            print(f"`is_instruct` set to {config['is_instruct']}")
        
        return config