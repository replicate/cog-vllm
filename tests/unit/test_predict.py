import sys
from pathlib import Path
import json
from unittest.mock import AsyncMock, MagicMock, patch, mock_open, Mock
import pytest

# Add the project root directory to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Mock required modules
mock_torch = MagicMock()
mock_torch.cuda.device_count.return_value = 1  # Set a default return value

sys.modules["torch"] = mock_torch
sys.modules["vllm"] = MagicMock()
sys.modules["vllm.engine"] = MagicMock()
sys.modules["vllm.engine.arg_utils"] = MagicMock()
sys.modules["vllm.sampling_params"] = MagicMock()


from predict import Predictor, PredictorConfig, UserError # pylint: disable=import-error, wrong-import-position


class MockInput: # pylint: disable=too-few-public-methods
    """
    Use this to mock default inputs for the Predictor class.
    """
    def __init__(self, default=None, **kwargs):
        self.default = default
        self.__dict__.update(kwargs)

    def __bool__(self):
        return bool(self.default)


sys.modules["cog"] = Mock()
sys.modules["cog"].Input = MockInput


@pytest.fixture
def mock_dependencies():
    with patch("predict.AsyncLLMEngine") as mock_engine_class, patch(
        "predict.AsyncEngineArgs"
    ) as mock_engine_args, patch(
        "predict.resolve_model_path"
    ) as mock_resolve_model_path, patch(
        "predict.torch", mock_torch
    ):  # Explicitly patch torch in predict.py

        mock_engine = AsyncMock()
        mock_engine_class.from_engine_args.return_value = mock_engine
        mock_resolve_model_path.return_value = "/path/to/weights"

        yield {
            "engine": mock_engine,
            "engine_class": mock_engine_class,
            "engine_args": mock_engine_args,
            "resolve_model_path": mock_resolve_model_path,
            "torch": mock_torch,
        }


@pytest.fixture
def mock_predictor_config():
    return {
        "prompt_template": "Test template: {prompt}",
        "engine_args": {"dtype": "float16", "tensor_parallel_size": 2},
    }


@pytest.mark.asyncio
async def test_setup_with_predictor_config(mock_dependencies, mock_predictor_config):
    with patch("builtins.open", mock_open(read_data=json.dumps(mock_predictor_config))):
        with patch("os.path.exists", return_value=True):
            predictor = Predictor()
            await predictor.setup("dummy_weights")

    assert isinstance(predictor.config, PredictorConfig)
    assert predictor.config.prompt_template == mock_predictor_config["prompt_template"]
    # assert predictor.config.engine_args == mock_predictor_config["engine_args"]

    mock_dependencies["engine_args"].assert_called_once_with(
        model="/path/to/weights", dtype="float16", tensor_parallel_size=2
    )


@pytest.mark.asyncio
async def test_setup_without_predictor_config(mock_dependencies):
    with patch("os.path.exists", return_value=False):
        predictor = Predictor()
        await predictor.setup("dummy_weights")

    assert isinstance(predictor.config, PredictorConfig)
    assert predictor.config.prompt_template is None
    assert predictor.config.engine_args == {}

    mock_dependencies["engine_args"].assert_called_once_with(
        model="/path/to/weights", dtype="auto", tensor_parallel_size=1
    )


@pytest.mark.asyncio
async def test_setup_with_invalid_predictor_config():
    invalid_config = {
        "prompt_template": 123,  # Should be a string
        "engine_args": "not a dict",  # Should be a dictionary
    }
    with patch("builtins.open", mock_open(read_data=json.dumps(invalid_config))):
        with patch("os.path.exists", return_value=True):
            with patch("predict.resolve_model_path", return_value="dummy_weights"):
                predictor = Predictor()
                with pytest.raises(UserError) as exc_info:
                    await predictor.setup("dummy_weights")
                assert "E1202 InvalidPredictorConfig:" in str(exc_info.value)

@pytest.mark.asyncio
async def test_predict(mock_dependencies):
    class MockResult: # pylint: disable=too-few-public-methods
        """
        Use this to mock the result object that the engine returns.
        """
        def __init__(self, text):
            self.outputs = [MagicMock(text=text)]

    # Define an async generator function
    async def mock_generate(*args, **kwargs): # pylint: disable=unused-argument
        yield MockResult("Generated text")

    mock_dependencies["engine"].generate = mock_generate

    predictor = Predictor()
    predictor.log = MagicMock()  # Mock the log method
    await predictor.setup("dummy_weights")

    # Mock the tokenizer
    predictor.tokenizer = MagicMock()
    predictor.tokenizer.chat_template = None
    predictor.tokenizer.eos_token_id = 0

    # Call the predict method
    result = predictor.predict(
        prompt="Test prompt", prompt_template=MockInput(default=None)
    )

    # Consume the async generator
    texts = []
    async for item in result:
        texts.append(item)

    assert texts == ["Generated text"]
