"""Tests for automatic model detection based on available API keys."""

import os
from unittest.mock import MagicMock, patch

import pytest

from langstruct.providers.llm_factory import LLMFactory


class TestAutoModelDetection:
    """Test suite for automatic model detection."""

    def test_detect_openai_key(self):
        """Test that OpenAI model is selected when OPENAI_API_KEY is set."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            model = LLMFactory.get_default_model()
            assert model == "gpt-5-mini"

    def test_detect_google_key(self):
        """Test that Google model is selected when GOOGLE_API_KEY is set."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True):
            model = LLMFactory.get_default_model()
            assert model == "gemini/gemini-2.5-flash"

    def test_detect_anthropic_key(self):
        """Test that Anthropic model is selected when ANTHROPIC_API_KEY is set."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            model = LLMFactory.get_default_model()
            assert model == "claude-3-5-haiku-latest"

    def test_priority_order(self):
        """Test that OpenAI is preferred when multiple keys are available."""
        # OpenAI should be preferred over Google
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-key", "GOOGLE_API_KEY": "test-key"},
            clear=True,
        ):
            model = LLMFactory.get_default_model()
            assert model == "gpt-5-mini"

        # Google should be preferred over Anthropic
        with patch.dict(
            os.environ,
            {"GOOGLE_API_KEY": "test-key", "ANTHROPIC_API_KEY": "test-key"},
            clear=True,
        ):
            model = LLMFactory.get_default_model()
            assert model == "gemini/gemini-2.5-flash"

    def test_ollama_detection(self):
        """Test that Ollama is detected when available and no API keys are set."""
        # Mock subprocess to simulate Ollama being available
        with patch.dict(os.environ, {}, clear=True):
            with patch("subprocess.run") as mock_run:
                # Simulate successful ollama list with llama3.1
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = "llama3.1:latest\nllama3:latest"
                mock_run.return_value = mock_result

                model = LLMFactory.get_default_model()
                assert model == "ollama/llama3.1"

    def test_ollama_fallback_models(self):
        """Test Ollama fallback model selection."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("subprocess.run") as mock_run:
                # Only llama3 available
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = "llama3:latest"
                mock_run.return_value = mock_result

                model = LLMFactory.get_default_model()
                assert model == "ollama/llama3"

    def test_no_api_keys_no_ollama(self):
        """Test fallback when no API keys are set and Ollama is not available."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("subprocess.run") as mock_run:
                # Simulate Ollama not being available
                mock_run.side_effect = FileNotFoundError()

                with pytest.warns(UserWarning, match="No API keys found"):
                    model = LLMFactory.get_default_model()
                    assert model == "gpt-5-mini"

    def test_ollama_timeout(self):
        """Test that timeout is handled gracefully when checking Ollama."""
        import subprocess

        with patch.dict(os.environ, {}, clear=True):
            with patch("subprocess.run") as mock_run:
                # Simulate timeout
                mock_run.side_effect = subprocess.TimeoutExpired("ollama", 2)

                with pytest.warns(UserWarning, match="No API keys found"):
                    model = LLMFactory.get_default_model()
                    assert model == "gpt-5-mini"


class TestLangStructAutoDefault:
    """Test that LangStruct uses auto-detection correctly."""

    def test_langstruct_uses_auto_detection(self):
        """Test that LangStruct uses auto-detected model when none specified."""
        import dspy

        from langstruct import LangStruct

        # Clear any existing DSPy configuration
        original_lm = getattr(dspy.settings, "lm", None)
        dspy.settings.lm = None

        try:
            # Set only Google API key to test auto-detection
            with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True):
                # Create a test LM that mimics real DSPy LM interface
                class TestLM:
                    def __init__(self, model="test-model"):
                        self.model = model
                        self.kwargs = {"model": model}
                        self._test_model_name = f"test-{model}"

                    def __call__(self, *args, **kwargs):
                        return MagicMock()

                with patch(
                    "langstruct.providers.llm_factory.LLMFactory.create_lm"
                ) as mock_create:
                    mock_lm = TestLM("gemini-flash")
                    mock_create.return_value = mock_lm

                    # Create LangStruct without specifying model
                    ls = LangStruct(example={"test": "value"})

                    # Verify create_lm was called with the auto-detected Google model
                    # (since only GOOGLE_API_KEY is set)
                    mock_create.assert_called()
                    call_args = mock_create.call_args[0]
                    assert call_args[0] == "gemini/gemini-2.5-flash"
        finally:
            # Restore original DSPy configuration
            if original_lm:
                dspy.settings.lm = original_lm

    def test_explicit_model_overrides_auto_detection(self):
        """Test that explicitly specified model overrides auto-detection."""
        from langstruct import LangStruct

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True):
            # Create a test LM that mimics real DSPy LM interface
            class TestLM:
                def __init__(self, model="test-model"):
                    self.model = model
                    self.kwargs = {"model": model}
                    self._test_model_name = f"test-{model}"

                def __call__(self, *args, **kwargs):
                    return MagicMock()

            with patch(
                "langstruct.providers.llm_factory.LLMFactory.get_default_model"
            ) as mock_get_default:
                with patch(
                    "langstruct.providers.llm_factory.LLMFactory.create_lm"
                ) as mock_create:
                    mock_lm = TestLM("gpt4o-mini")
                    mock_create.return_value = mock_lm

                    # Create LangStruct with explicit model
                    ls = LangStruct(example={"test": "value"}, model="gpt-4o")

                    # Verify get_default_model was NOT called
                    mock_get_default.assert_not_called()

                    # Verify create_lm was called with the explicit model
                    mock_create.assert_called()
                    call_args = mock_create.call_args[0]
                    assert call_args[0] == "gpt-4o"
