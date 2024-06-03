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

```console
# Clone the cog-vllm repository
$ git clone https://github.com/replicate/cog-vllm.git
$ cd cog-vllm

# Make a single prediction
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

```console
$ cog login

$ cog push r8.im/<your-username>/<your-model-name>
--> ...
--> Pushing image 'r8.im/...'
```

[Replicate]: https://replicate.com
[vLLM-supported language model]: https://docs.vllm.ai/en/latest/models/supported_models.html
