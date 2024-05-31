# Cog-vLLM: Run vLLM on Replicate

Cog-vLLM is a [Cog](https://github.com/replicate/cog)-compatible vLLM server. 

vLLM is a fast and easy-to-use library for LLM inference and serving. Cog is an open-source tool that lets you package machine learning models and servers in a standard, production-ready container. 


With Cog-vLLM, you can easily deploy production-grade LLMs to your own infrastructure, or to [Replicate](https://replicate.com/).


# Highlights

* :rocket: **Run any vLLM-supported language model on Replicate.** Deploying language models can be a pain, particularly if you care about performance. With Cog-vLLM, all you have to do is specify a model and call `cog push`! 

* :trident: **Configurable support for continuous batching.** Cog-vLLM leverage's Cog's experimental support for asynchronous processes, which means that continuous batching works out of the box. 

* :unlock: **Open source, all the way down**. You can make Cog-vLLM yours. Hack, modify, contribute, open it up and peer inside—make it do exactly what you need.


# Quickstart

## Build a Cog-vLLM image
If you want to worry about the details later, you can call the `cog-vllm-helper` script. Just pass a model-id and the URL to a flat tarball that contains your model assets.

The following commands will generate a `config.yaml` for Cog-vLLM and build a Cog-vLLM image.

```console
$ python cog-vllm-helper.py \
  --model-id mistralai/mistral-7b-instruct-v0.2 \
  --model-url https://weights.replicate.delivery/default/official-models/hf/mistralai/mistral-7b-instruct-v0.2/model.tar
--> ...
--> Generated config.yaml:
--> -------------------------
--> {'model_id': 'mistralai/mistral-7b-instruct-v0.2',
--> 'model_url': 'https://weights.replicate.delivery/default/official-models/hf/mistralai/mistral-7b-instruct-v0.2/model.tar'}

## Make a local prediction with your image

If you don't have a local GPU, you can go ahead and push your image to Replicate (see the next step). If you do have a local GPU, run:

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

## Push to Replicate

Finally, you can push your Cog-vLLM model to Replicate. To do this, auth to Replicate, set your API key and, if it doesn't already exist, create the Replicate model that you want to push to.

You can create your model on [Replicate web](https://replicate.com/create) or you can use the [Replicate CLI](https://github.com/replicate/cli). 

After you create your model, run:

```console
$ cog login
$ export REPLICATE_API_TOKEN=<your token>
$ cog push r8.im/<your user name>/<your model name>
--> ...
--> Pushing image 'r8.im/...'
$ prediction_id=$(curl -s -X POST \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "version": <your model version>,
    "input": {
      "prompt": "Hello!"
    }
  }' \
  https://api.replicate.com/v1/predictions | jq -r '.id')

$ curl -s \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  "https://api.replicate.com/v1/predictions/${prediction_id}"
```

# How it works

Cog-vLLM relies on `config.yaml` to setup the server and intialize a vLLM `AsyncEngine`. The format of `config.yaml` is:

```yaml
# Note: The ID of the model you want to run. 
model_id: mistralai/mistral-7b-instruct-v0.2
# Note: A URL to a flat tarball containing all model assets
model_url: https://weights.replicate.delivery/default/official-models/hf/mistralai/mistral-7b-instruct-v0.2/model.tar
# Note: A bool that specifies if your model is an instruct/chat model. 
#       Cog-vLLM will attempt to infer this if you do not specify it.
#       If `is_instruct` is `False`, `system_prompt` will be removed from your model schema.
is_instruct: ...     # Optional
# Note: You can specify a default prompt template here.
#       If you don't specify one, Cog-vLLM will attempt to infer one, e.g. see `./prompte_templates.py`
prompt_template: ... # Optional
# Note: You can specify a default system prompt here.
#       If you don't specify one, `"You are a helpful assistent."` will be used.
system_prompt: ...   # Optional
# Note: You can specify any valid `vllm.engine.arg_utils.AsyncEngineArgs` arguments. 
#       These will be used to initialize the `AsyncEngine`.
engine_args: ...     # Optional
  ...
```

### Example

Copy and paste the following yaml into `config.yaml`:

```yaml
model_id: mistralai/mistral-7b-instruct-v0.2
model_url: https://weights.replicate.delivery/default/official-models/hf/mistralai/mistral-7b-instruct-v0.2/model.tar
system_prompt: "You are a helpful assistant that always speaks like a 13th century Knight."
engine_args:
   tensor_parallel_size: 1
```

Now, if you have a GPU available, you can run predictions on this model:

```console
$ cog predict -i prompt="Write an epic poem about open source machine learning."
--> Building Docker image...
--> Running Prediction...
--> [
-->  " In",
-->  " the",
-->  " realm",
-->  " of",
-->  " knowledge",
-->  ",",
-->  " where",
-->  " the",
-->  " wise",
-->  " ones",
-->  " dwell",
-->  ",",
-->  "\n",
-->  "A",
-->  " tale",
-->  " of",
-->  " valor",
-->  " and",
-->  " of",
-->  " learning",
-->  ",",
-->  " I",
-->  " tell",
-->  ".",
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
    -d '{"input": {"prompt": "Hello!"}}'
```

### Push to replicate


```console
$ cog login
$ export REPLICATE_API_TOKEN=<your token>
$ cog push r8.im/<your user name>/<your model name>
--> ...
--> Pushing image 'r8.im/...'
```

# Debug and develop in a cog-vllm container

```console
$ cog run -p 5000 /bin/bash
--> Running '/bin/bash' in Docker with the current directory mounted as a volume...
$ python -m cog.server.http
```

Then, in a different terminal session, you can make a request against your local server:

```console
$ curl http://localhost:5000/predictions -X POST \
    -H 'Content-Type: application/json' \
    -d '{"input": {"prompt": "Hello!"}}'
```

