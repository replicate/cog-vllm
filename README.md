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

Go to [replicate.com/replicate/vllm](https://replicate.com/replicate/vllm)
and create a new vLLM model from a [supported Hugging Face repo][vLLM-supported language model],
such as [google/gemma-2b](https://huggingface.co/google/gemma-2b)

> [!IMPORTANT]  
> Gated models require a [Hugging Face API token](https://huggingface.co/settings/tokens),
> which you can set in the `hf_token` field of the model creation form.

<img width="1055" alt="Create a new vLLM model on Replicate" src="https://github.com/replicate/cog-vllm/assets/7659/a8f31837-0ed3-40f7-974c-d0a16ae48350">

Replicate downloads the model files, packages them into a `.tar` archive,
and pushes a new version of your model that's ready to use.

<img width="1322" alt="Trained vLLM model on Replicate" src="https://github.com/replicate/cog-vllm/assets/7659/ebb84e12-9173-4fb0-8749-7293a105cf13">

From here, you can either use your model as-is,
or customize it and push up your changes.

## Local Development

If you're on a machine or VM with a GPU,
you can try out changes before pushing them to Replicate.

Start by [installing or upgrading Cog](https://cog.run/#install).
You'll need Cog [v0.10.0-alpha11](https://github.com/replicate/cog/releases/tag/v0.10.0-alpha11):

```console
$ sudo curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/download/v0.10.0-alpha11/cog_$(uname -s)_$(uname -m)"
$ sudo chmod +x /usr/local/bin/cog
```

Then clone this repository:

```console
$ git clone https://github.com/replicate/cog-vllm
$ cd cog-vllm
```

Go to the [Replicate dashboard](https://replicate.com/trainings) and 
navigate to the training for your vLLM model.
From that page, copy the weights URL from the <kbd>Download weights</kbd> button.

<img width="642" alt="Copy weights URL from Replicate training" src="https://github.com/replicate/cog-vllm/assets/7659/97c403a9-ec49-418a-a7e2-b37cb0e0bb8c">

Set the `COG_WEIGHTS` environment variable with that copied value: 

```console
$ export COG_WEIGHTS="..."
```

Now, make your first prediction against the model locally:

```console
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
