import subprocess


def test_predict():
    # Mirror of https://huggingface.co/EleutherAI/pythia-70m
    # pylint: disable=line-too-long
    weights_url = "https://replicate.delivery/czjl/HUTgHv0M6FbnJxzkbe7Ly1fN19tabwYOZTFLuJld3f7MifpLB/model.tar"

    # Run the cog predict command and capture the output
    result = subprocess.run(
        ["cog", "predict", "-e", f"COG_WEIGHTS={weights_url}", "-i", "prompt=Hello!"],
        capture_output=True,
        text=True,
        check=True,
    )

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
