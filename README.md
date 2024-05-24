# Cog-vLLM: Run vLLM on Replicate

Cog-vLLM is a [Cog](https://github.com/replicate/cog)-compatible vLLM server. 

vLLM is a fast and easy-to-use library for LLM inference and serving. Cog is an open-source tool that lets you package machine learning models and servers in a standard, production-ready container. 


With Cog-vLLM, you can easily deploy production-grade LLMs to your own infrastructure, or to [Replicate](https://replicate.com/).


# Highlights

* :rocket: **Run any vLLM-supported language model on Replicate.** Deploying language models can be a pain, particularly if you care about performance. With Cog-vLLM, all you have to do is specify a model and call `cog push`! 

* :trident: **Configurable support for continuous batching.** Cog-vLLM leverage's Cog's experimental support for asynchronous processes, which means that continuous batching works out of the box. 

* :unlock: **Open source, all the way down**. You can make Cog-vLLM yours. Hack, modify, contribute, open it up and peer inside—make it do exactly what you need.


# Quickstart

If you want to worry about the details later, you can call the `cog-vllm-helper` script. Just pass a model-id and the URL to a flat tarball that contains your model assets.

If you have a local GPU, the commands below will download and run `mistral-7b-instruct-v0.2`.

```console
$ python cog-vllm-helper.py \
  --model-id mistralai/mistral-7b-instruct-v0.2 \
  --model-url https://weights.replicate.delivery/default/official-models/hf/mistralai/mistral-7b-instruct-v0.2/model.tar
--> ...
--> Generated config.yaml:
--> -------------------------
--> {'model_id': 'mistralai/mistral-7b-instruct-v0.2',
--> 'model_url': 'https://weights.replicate.delivery/default/official-models/hf/mistralai/mistral-7b-instruct-v0.2/model.tar'}

$ cog predict -i prompt="Hello!"
--> Starting Docker image cog-vllm-base and running setup()...
--> Formatted prompt: <s>[INST] You are a helpful assistant. Hello! [/INST]
--> [
-->  " Hello",
-->  "!",
-->  " I",
-->  "'",
-->  "m",
-->  " here",
-->  " to",
-->  " help",
-->  " answer",
-->  " any",
-->  " questions",
-->  " you",
-->  " might",
-->  " have",
--> ...
--> ]
```

# How it works

Cog-vLLM relies on `config.yaml` to setup the server and intialize a vLLM `AsyncEngine`. The format of `config.yaml` is:

```yaml
# Note: The ID of the model you want to run. 
model_id: mistralai/mistral-7b-instruct-v0.2
# Note: A URL to a flat tarball containing all model assets
model_url: https://weights.replicate.delivery/default/official-models/hf/mistralai/mistral-7b-instruct-v0.2/model.tar
# 
is_instruct: ...     # Optional
prompt_template: ... # Optional
system_prompt: ...   # Optional
engine_args: ...     # Optional
  ...
```

In `config.yaml`, specify a model ID and the remote path to a flat tarball that contains your model assets (e.g. weights, configs, tokenizer, etc). 

```yaml
model_id: mistralai/mistral-7b-instruct-v0.2
model_url: https://weights.replicate.delivery/default/official-models/hf/mistralai/mistral-7b-instruct-v0.2/model.tar
```

Now, if you have a GPU available, you can run predictions on this model:

```console
$ cog predict -i prompt="Write an epic poem about open source machine learning."
--> Building Docker image...
--> Running Prediction...
--> [
-->  "\n",
-->  "\n",
-->  "In",
-->  " the",
-->  " land",
-->  " of",
-->  " code",
-->  ",",
-->  " where",
-->  " secrets",
-->  " blo",
-->  "om",
-->  ",",
-->  ...
-->  ]
```

Or, build a Docker image for deployment:

```console
$ cog build -t cog-vllm
--> Building Docker image...
--> Built cog-vllm:latest

$ docker run -d -p 5000:5000 --gpus all cog-vllm

$ curl http://localhost:5000/predictions -X POST \
    -H 'Content-Type: application/json' \
    -d '{"input": {"image": "https://.../input.jpg"}}'
```
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
