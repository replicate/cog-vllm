#!/usr/bin/env python3
import os
import subprocess
import click
import platform


def update_env_file(model_id, cog_weights):
    env_file = ".env"
    with open(env_file, "r") as file:
        lines = file.readlines()

    updated_lines = []
    for line in lines:
        if model_id:
            if line.startswith("MODEL_ID="):
                line = f"MODEL_ID={model_id}\n"
        if cog_weights:
            if line.startswith("COG_WEIGHTS="):
                line = f'COG_WEIGHTS="{cog_weights}"\n'
        updated_lines.append(line)

    # if MODEL_ID and COG_WEIGHTS lines were missing add them
    if not any(line.startswith("MODEL_ID=") for line in updated_lines):
        updated_lines.append(f"MODEL_ID={model_id}\n")
    if not any(line.startswith("COG_WEIGHTS=") for line in updated_lines):
        updated_lines.append(f'COG_WEIGHTS="{cog_weights}"\n')

    with open(env_file, "w") as file:
        file.writelines(updated_lines)


@click.command()
@click.option(
    "-m",
    "--model-id",
    default=None,
    help="Specify the model ID to use (e.g., mistralai/Mistral-7B-Instruct-v0.2)",
)
@click.option(
    "-w",
    "--cog-weights",
    default=None,
    help="Specify the URL for the COG weights (e.g., https://weights.replicate.delivery/default/mistral-7b-instruct-v0.2)",
)
@click.option(
    "-t",
    "--tag-name",
    default="cog-vllm",
    help="Specify the tag name for the COG build (e.g., my-custom-tag)",
)
def main(model_id, cog_weights, tag_name):
    # create .env file if it doesn't exist
    if not os.path.exists(".env"):
        os.system("touch .env")
    update_env_file(model_id, cog_weights)
    # Install COG
    cog_url = f"https://github.com/replicate/cog/releases/download/v0.10.0-alpha5/cog_{platform.system().lower()}_{platform.machine().lower()}"
    subprocess.run(["sudo", "curl", "-o", "/usr/local/bin/cog", "-L", cog_url])
    subprocess.run(["sudo", "chmod", "+x", "/usr/local/bin/cog"])

    # Build with COG using the specified tag name
    subprocess.run(["cog", "build", "-t", tag_name])


if __name__ == "__main__":
    main()
