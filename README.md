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

## Getting Started

If you're on a machine or VM with a GPU,
you can run vLLM models locally using this cog-vllm wrapper.

### Prerequisites

You'll need:
- Either a huggingface model (username/model-name) or a URL to a `.tar` archive containing your model weights
- Cog installed ([v0.16.8](https://github.com/replicate/cog/releases/tag/v0.16.8) or newer)
- A GPU-enabled machine for optimal performance

### Installation

Start by [installing or upgrading Cog](https://cog.run/#install):

```console
$ sudo curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/download/v0.16.8/cog_$(uname -s)_$(uname -m)"
$ sudo chmod +x /usr/local/bin/cog
```

Then clone this repository:

```console
$ git clone https://github.com/replicate/cog-vllm
$ cd cog-vllm
```

Set the `COG_WEIGHTS` variable in `predict.py` with your model weights model name or URL:

```console
$ COG_WEIGHTS="https://your-weights-url.com/model.tar"
# or for local development:
$ COG_WEIGHTS="Qwen/Qwen3-VL-8B-Instruct"
```

## To create a model.tar file:

```console
$ huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir checkpoints
$ cd checkpoints
$ tar -cvf model.tar --exclude=model.tar -C . .
```
Then upload model.tar to your own storage service.

### Running Predictions

Make your first prediction against the model locally:

```console
$ cog predict -i prompt="Who are you?"
```

The first time you run this command with a model name or URL,
Cog downloads the model weights and saves them to a local checkpointsdirectory.

## Deploying to Replicate

When you're ready to deploy your model to Replicate,
you can push your changes:


```shell
$ cog login
$ cog push r8.im/<your-username>/<model-name>
--> ...
--> Pushing image 'r8.im/...'
```

### Using Your Model on Replicate

After you push your model, you can run it via the Replicate API.

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
    input={"prompt": "Who are you?"},
    stream=True
)

for event in prediction.stream():
    print(str(event), end="")
```

[Replicate]: https://replicate.com
[replicate-python]: https://github.com/replicate/replicate-python
