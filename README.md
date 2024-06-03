# Cog-vLLM: Run vLLM on Replicate

[Cog](https://github.com/replicate/cog) 
is an open-source tool that lets you package machine learning models
in a standard, production-ready container. 
vLLM is a fast and easy-to-use library for LLM inference and serving.

You can deploy your packaged model to your own infrastructure, 
or to [Replicate].

## Highlights

* 🚀 **Run vLLM in the cloud with an API**.
  Deploy any [vLLM-supported language model] at scale on Replicate.

* 🏭 **Support multiple concurrent requests**.
  Continuous batching works out of the box.

* 🐢 **Open Source, all the way down**.
  Look inside, take it apart, make it do exactly what you need.

## Quickstart

Go to [replicate.com/vllm](https://replicate.com/vllm)
and create a new vLLM model from a Hugging Face repo.

Replicate downloads the model files, packages them into a `.tar` archive,
and pushes a new version of your model that's ready to use.

From here, you can either use your model as-is,
or customize it and push up your changes.

## Local Development

If you're on a machine or VM with a GPU,
you can try out changes before pushing them to Replicate.

Start by [installing or upgrading Cog](https://cog.run/#install).
You'll need Cog v0.9.9 or later:

```console
$ brew install cog
```

Then clone this repository:

```console
$ git clone https://github.com/replicate/cog-vllm
$ cd cog-vllm
```

Make your first prediction against the model:

```console
$ export COG_WEIGHTS="..." # copy the URL to "Download Weights" from Replicate
$ cog predict -e "COG_WEIGHTS=$COG_WEIGHTS" \ 
              -i prompt="Hello!"
```

The first time you run this command,
Cog downloads the model weights and save them to the `models` subdirectory.

To make multiple predictions,
start up the HTTP server and send it `POST /predictions` requests.

```console
# Start the HTTP server
$ cog run -p 5000 -e "COG_WEIGHTS=$COG_WEIGHTS" python -m cog.server.http

# In a different terminal session, send requests to the server
$ curl http://localhost:5000/predictions -X POST \
    -H 'Content-Type: application/json' \
    -d '{"input": {"prompt": "Hello!"}}'
```

When you're finished working,
you can push your changes to Replicate.

Grab your token from [replicate.com/account](https://replicate.com/account) 
and set it as an environment variable:

```shell
export REPLICATE_API_TOKEN=<your token>
```

```console
$ echo $REPLICATE_API_TOKEN | cog login --token-stdin
$ cog push r8.im/<your-username>/<your-model-name>
--> ...
--> Pushing image 'r8.im/...'
```

After you push your model, you can try running it on Replicate.

Install the [Replicate Python SDK][replicate-python]:

```console
$ pip install replicate
```

Create a prediction and stream its output:

```python
import replicate

model = replicate.models.get("<your-username>/<your-model-name>")
prediction = replicate.predictions.create(
    version=model.latest_version,
    input={ "prompt": "Hello" },
    stream=True
)

for event in prediction.stream():
    print(str(event), end="")
```

[Replicate]: https://replicate.com
[vLLM-supported language model]: https://docs.vllm.ai/en/latest/models/supported_models.html
[replicate-python]: https://github.com/replicate/replicate-python
