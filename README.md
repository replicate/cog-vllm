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

* 🔄 **Async streaming support**.
  Built with async/await for efficient streaming responses.

* 🐢 **Open Source, all the way down**.
  Look inside, take it apart, make it do exactly what you need.

## Repository Structure

```
cog-vllm/
├── predict.py              # Main prediction logic with vLLM integration
├── train.py               # Training/fine-tuning support
├── utils.py               # Utility functions for model handling
├── prompt_templates.py    # Prompt templates for different model types
├── cog.yaml              # Cog configuration
├── requirements.txt      # Python dependencies
├── test.sh              # Test script
├── tests/               # Test suites
│   ├── unit/           # Unit tests
│   └── end_to_end/     # End-to-end tests
└── README.md           # This file
```

## Features

- **Latest vLLM Integration**: Uses vLLM 0.9.2 with AsyncLLMEngine for optimal performance
- **Latest Cog Support**: Updated for modern Cog features including async predictors
- **Flexible Prompt Templates**: Support for custom prompt templates and chat formats
- **Comprehensive Configuration**: Configurable via `predictor_config.json`
- **Streaming Support**: Real-time token streaming with AsyncConcatenateIterator
- **GPU Optimized**: CUDA 12.6 support with multi-GPU tensor parallelism

## Quickstart

Go to [replicate.com/replicate/vllm](https://replicate.com/replicate/vllm)
and create a new vLLM model from a [supported Hugging Face repo][vLLM-supported language model],
such as [google/gemma-2b](https://huggingface.co/google/gemma-2b)

> [!IMPORTANT]  
> Gated models require a [Hugging Face API token](https://huggingface.co/settings/tokens),
> which you can set in the `hf_token` field of the model creation form.

## Local Development

If you're on a machine or VM with a GPU,
you can try out changes before pushing them to Replicate.

Start by [installing Cog](https://cog.run/#install):

```console
$ curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"
$ sudo chmod +x /usr/local/bin/cog
$ sudo mv /usr/local/bin/cog /usr/local/bin/
```

Then clone this repository:

```console
$ git clone https://github.com/replicate/cog-vllm
$ cd cog-vllm
```

Set the model weights URL as an environment variable:

```console
$ export COG_WEIGHTS="https://huggingface.co/microsoft/DialoGPT-medium"
```

Now, make your first prediction against the model locally:

```console
$ cog predict -e "COG_WEIGHTS=$COG_WEIGHTS" -i prompt="Hello!"
```

### Development Server

To make multiple predictions, start up the HTTP server:

```console
# Start the HTTP server
$ cog run -p 5000 -e "COG_WEIGHTS=$COG_WEIGHTS" python -m cog.server.http

# In a different terminal session, send requests
$ curl http://localhost:5000/predictions -X POST \
    -H 'Content-Type: application/json' \
    -d '{"input": {"prompt": "Hello!", "max_tokens": 50}}'
```

### Configuration

You can customize the model behavior by creating a `predictor_config.json` file:

```json
{
  "prompt_template": "### User: {prompt}\n### Assistant: ",
  "engine_args": {
    "tensor_parallel_size": 2,
    "dtype": "bfloat16",
    "max_model_len": 4096
  }
}
```

### Testing

Run the test suite:

```console
$ ./test.sh
```

Or run specific tests:

```console
$ python -m pytest tests/unit/
$ python -m pytest tests/end_to_end/
```

## Dependencies

The current implementation uses:

- **vLLM 0.9.2**: Latest version with improved async support
- **PyTorch 2.7.0**: For GPU acceleration
- **Jinja2 3.1.6**: For prompt templating
- **CUDA 12.6**: For optimal GPU performance

See `requirements.txt` for the complete dependency list.

## Deployment

When you're ready to deploy:

```console
$ export REPLICATE_API_TOKEN=<your-token>
$ echo $REPLICATE_API_TOKEN | cog login --token-stdin
$ cog push r8.im/<your-username>/<your-model-name>
```

## Usage Examples

### Basic Text Generation

```python
import replicate

model = replicate.models.get("<your-username>/<your-model-name>")
prediction = replicate.predictions.create(
    version=model.latest_version,
    input={
        "prompt": "Explain quantum computing",
        "max_tokens": 200,
        "temperature": 0.7
    },
    stream=True
)

for event in prediction.stream():
    print(str(event), end="")
```

### Chat Format

```python
prediction = replicate.predictions.create(
    version=model.latest_version,
    input={
        "prompt": "What is the capital of France?",
        "system_prompt": "You are a helpful geography expert.",
        "max_tokens": 100
    }
)
```

[Replicate]: https://replicate.com
[vLLM-supported language model]: https://docs.vllm.ai/en/latest/models/supported_models.html
