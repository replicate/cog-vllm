
# Setup

First prepare the environment.  The input parameters are set in the .env file.  You can change the default values there.  You can overide two environment variables `MODEL_ID` and `COG_WEIGHTS` by passing the `--m` and `--c` flags to the script.

```
$ ./setup.sh -h

Usage: setup.sh [-h] [-m MODEL_ID] [-w COG_WEIGHTS] [-t TAG_NAME]

This script configures the environment for COG VLLM by optionally updating
MODEL_ID and COG_WEIGHTS in the .env file, installing COG, and allowing
a custom tag name for the COG build.

    -h          display this help and exit
    -m MODEL_ID specify the model ID to use (e.g., mistralai/Mistral-7B-Instruct-v0.2)
    -w COG_WEIGHTS specify the URL for the COG weights (e.g., https://weights.replicate.delivery/default/mistral-7b-instruct-v0.2)
    -t TAG_NAME  specify the tag name for the COG build (e.g., my-custom-tag)
```

This will build the cog container and tag it as `cog-vllm`, unless you override that with the `-t` flag.

For example to build with mistral-7b-instruct-v0.2 you would run:

```bash
./setup.sh -m mistralai/Mistral-7B-Instruct-v0.2 \
           -w https://weights.replicate.delivery/default/mistral-7b-instruct-v0.2
```


# Push to replicate

```bash
cog login
cog push <replicate destination> # r8.im/hamelsmu/test-mistral-7b-instruct-v0.2
```


# Debug The Container

```bash
cog run -e CUDA_VISIBLE_DEVICES=0 -p 5000 /bin/bash
# in the container run this for debugging
python predict.py
```

or run `cog predict`:

```bash
cog predict -e CUDA_VISIBLE_DEVICES=0 \
           -i prompt="Write a blogpost about SEO directed at a technical audience" \
           -i max_new_tokens=512 \
           -i temperature=0.6 \
           -i top_p=0.9 \
           -i top_k=50 \
           -i presence_penalty=0.0 \
           -i frequency_penalty=0.0 \
           -i prompt_template="<s>[INST] {prompt} [/INST] "
```
