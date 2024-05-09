import asyncio
import os
import typing as tp


def check_files_exist(remote_files: list[str], local_path: str) -> list[str]:
    local_files = os.listdir(local_path)
    missing_files = list(set(remote_files) - set(local_files))
    return missing_files


async def download_files_with_pget(
    remote_path: str, path: str, files: list[str]
) -> None:
    import pdb

    pdb.set_trace()
    # get all the files that are not .tars
    download_jobs = "\n".join(
        f"{remote_path}/{f} {path}/{f}" for f in files if not f.endswith(".tar")
    )
    # seperate jobs for files that end with .tar
    tar_download_jobs = [
        f"{remote_path}/{f}.tar ./{path}/{f}" for f in files if f.endswith(".tar")
    ]
    args = ["pget", "multifile", "-", "-f", "--max-conn-per-host", "100"]
    process = await asyncio.create_subprocess_exec(*args, stdin=-1, close_fds=True)
    # Wait for the subprocess to finish
    await process.communicate(download_jobs.encode())
    # download tars
    for tar_job in tar_download_jobs:
        # run pget -x -f remote_path/f.tar path/f
        args = ["pget", "-x", "-f", tar_job]
        process = await asyncio.create_subprocess_exec(*args, close_fds=True)


async def maybe_download_with_pget(
    path: str,
    remote_path: tp.Optional[str] = None,
    remote_filenames: tp.Optional[list[str]] = None,
):
    if remote_path:
        remote_path = remote_path.rstrip("/")
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            missing_files = remote_filenames or []
        else:
            missing_files = check_files_exist(remote_filenames or [], path)
        await download_files_with_pget(remote_path, path, missing_files)
    return path
