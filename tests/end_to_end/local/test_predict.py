import subprocess
import json
import os
import shutil


def test_predict():
    # Mirror of https://huggingface.co/EleutherAI/pythia-70m
    # pylint: disable=line-too-long
    predictor_config = {
        "engine_args": {"enforce_eager": True},
    }

    config_filename = "predictor_config.json"
    backup_filename = "predictor_config.json.bak"

    if os.path.exists(config_filename):
        shutil.move(config_filename, backup_filename)

    try:

        with open(config_filename, "w", encoding="utf-8") as temp_config:
            json.dump(predictor_config, temp_config, indent=4)
        weights_url = "https://weights.replicate.delivery/default/internal-testing/EleutherAI/pythia-70m/model.tar"  # pylint: disable=line-too-long

        result = subprocess.run(
            [
                "cog",
                "predict",
                "-e",
                f"COG_WEIGHTS={weights_url}",
                "-i",
                "prompt=Hello!",
                "-i",
                "max_tokens=10",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

    finally:
        os.remove(config_filename)
        if os.path.exists(backup_filename):
            shutil.move(backup_filename, config_filename)

    # Check that the cog predict command completed successfully
    assert result.returncode == 0, f"Cog predict failed with error: {result.stderr}"

    # Parse the output
    output = result.stdout.strip().splitlines()

    # Make assertions based on the expected output
    assert isinstance(output, list), "Output is not a list of strings"
    assert len(output) > 0, "Output list is empty"
    for line in output:
        assert isinstance(line, str), f"Output contains a non-string element: {line}"

    # Optionally print the output for debugging
    print("Output from cog predict:", output)
