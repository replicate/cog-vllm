import subprocess


def test_mistral():
    weights_url = "https://weights.replicate.delivery/default/official-models/hf/mistralai/mistral-7b-instruct-v0.2/model.tar"

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
