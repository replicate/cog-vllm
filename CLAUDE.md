# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Cog wrapper for vLLM that enables deploying vLLM-supported language models on Replicate. The project packages vLLM models in production-ready containers that can run locally or be deployed to Replicate's infrastructure.

## Core Architecture

### Main Components

- **predict.py**: The core Predictor class that:
  - Inherits from `cog.BasePredictor`
  - Uses vLLM's `AsyncLLMEngine` for inference with continuous batching
  - Implements async `setup()` and `predict()` methods
  - Loads model weights from URLs (tarballs) or local paths
  - Handles prompt formatting via templates or tokenizer chat templates
  - Supports configuration via `predictor_config.json`

- **utils.py**: Model weight management with `resolve_model_path()` that:
  - Downloads model tarballs using `pget` (parallel downloader)
  - Uses `/weights` volume for caching when available
  - Falls back to local directory if volume unavailable
  - Creates symlinks to optimize storage

- **prompt_templates.py**: Pre-defined prompt templates (COMPLETION, LLAMA_3_INSTRUCT, LLAMA_2_INSTRUCT, MISTRAL_INSTRUCT)

### Configuration System

The predictor can be configured via `predictor_config.json` with:
- `prompt_template`: Custom Jinja2 chat template (overrides tokenizer's template)
- `engine_args`: Dictionary passed directly to vLLM's `AsyncEngineArgs` (e.g., `max_model_len`, `enforce_eager`, `tensor_parallel_size`)

Priority order for prompt formatting:
1. Runtime `prompt_template` parameter
2. `predictor_config.json` prompt_template
3. Tokenizer's built-in chat template
4. Default COMPLETION template from prompt_templates.py

### Model Loading Flow

1. `setup()` receives weights path/URL (from `COG_WEIGHTS` env var)
2. `_extract_weights_path()` handles various Cog types (URLFile, URLPath, str) and checks for cached models
3. If URL is provided, checks `./checkpoints/{model_name}` for cached version first
4. `setup()` explicitly checks if returned path is local (cached) - if so, uses directly without download
5. If URL, `resolve_model_path()` downloads tarball to `./checkpoints/` (with additional cache check)
6. `load_config()` searches for `predictor_config.json` in weights dir, then current dir
7. AsyncLLMEngine initialized with engine_args (auto-detects tensor_parallel_size from GPU count)
8. AutoTokenizer loaded from model directory
9. Test prediction runs to warm up engine

### Local Model Caching

Models are automatically cached in `./checkpoints/{model_name}/` to avoid re-downloading with **three layers of protection**:

- **First run**: Model is downloaded from URL and extracted to `./checkpoints/Qwen3-VL-8B-Instruct/` (or other model name)
- **Subsequent runs**: Cached model is detected and used automatically (no download, no async operations)
- **Fallback**: If `/weights` volume is available (in production), it's used with symlinks
- **Manual cache**: You can pre-populate `./checkpoints/` with extracted model directories

The triple-layer caching logic:
1. `predict.py:_extract_weights_path()` - **First check**: Returns cached path if exists, avoids download path entirely
2. `predict.py:setup()` - **Second check**: Explicitly verifies local path and skips `resolve_model_path()` for cached models
3. `utils.py:maybe_download_tarball_with_pget()` - **Third check**: Safety net that verifies cache before any download operation

This ensures models are **NEVER re-downloaded** once cached, even across multiple container runs.

## Common Commands

### Development

```bash
# Run a single prediction locally
cog predict -e "COG_WEIGHTS=<url-or-path>" -i prompt="Hello!"

# Start HTTP server for multiple predictions
cog run -p 5000 -e "COG_WEIGHTS=<url>" python -m cog.server.http

# Run end-to-end tests
./test.sh

# Run tests with pytest
pytest tests/end_to_end/local/test_predict.py
pytest tests/unit
```

### Building & Deployment

```bash
# Build the container
cog build

# Push to Replicate
cog login --token-stdin
cog push r8.im/<username>/<model-name>
```

### Dependency Management

Dependencies are managed via pyproject.toml and compiled with uv:

```bash
# Update requirements.txt from pyproject.toml
uv pip compile pyproject.toml -o requirements.txt --python-version 3.13
```

Note: vLLM itself is installed via `cog.yaml` build step (not in requirements.txt) with specific CUDA arch flags.

## Key Technical Details

- **Python version**: 3.13
- **CUDA**: 12.4 with specific architectures (8.0, 8.6, 9.0)
- **vLLM version**: 0.11.0 (installed during build)
- **Torch version**: 2.9.0
- **Concurrency**: Max 32 concurrent requests (configured in cog.yaml)
- **Async throughout**: Uses async/await for setup and prediction
- **Error codes**: Custom error codes (E1000-E1004, E1200-E1202) for user-facing errors

### cuDNN Library Path Fix

vLLM 0.11.0 requires correct LD_LIBRARY_PATH settings to find cuDNN/cuBLAS libraries. The predictor automatically sets this at import time (predict.py:3-30) by:
1. Importing nvidia.cudnn.lib and nvidia.cublas.lib
2. Detecting library paths using __path__ or __file__ attributes
3. Prepending these paths to LD_LIBRARY_PATH before importing vLLM

This prevents the "undefined symbol: cudnnGetLibConfig" error that occurs with mismatched library paths.

## Testing Notes

Tests use `predictor_config.json` to configure engine behavior. The test temporarily creates/modifies this file to set `enforce_eager: true` for deterministic testing without CUDA graphs.
