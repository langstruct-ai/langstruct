"""LLM provider factory for creating DSPy language model instances."""

import os
import warnings
from typing import Any, Dict, Optional

import dspy


class LLMFactory:
    """Factory for creating DSPy language model instances.

    This factory provides a thin wrapper around DSPy's LM class,
    allowing any model supported by DSPy/LiteLLM to work automatically.
    """

    @classmethod
    def create_lm(
        cls,
        model_name: str,
        base_url: Optional[str] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> dspy.LM:
        """Create a DSPy language model instance from a model name.

        This method passes model names directly to DSPy/LiteLLM,
        which handles all provider-specific logic and model routing.

        Args:
            model_name: Any model name supported by DSPy/LiteLLM
                       Examples: "gpt-5-pro", "claude-opus-4-1", "gemini/gemini-pro",
                       "ollama/llama3", or any LiteLLM-compatible string
            base_url: Base URL for API calls (for custom deployments)
            max_tokens: Maximum tokens for response (default: 20_000 for reasoning compliance)
            **kwargs: Additional arguments to pass to the LM constructor

        Returns:
            DSPy LM instance

        Raises:
            Exception: Any errors from DSPy/LiteLLM during model initialization
        """
        # Set defaults that keep reasoning models happy while remaining overrideable
        if max_tokens is None:
            max_tokens = 20_000

        # Build configuration
        config = {"model": model_name, "max_tokens": max_tokens, **kwargs}
        config.setdefault("temperature", 1.0)

        if base_url:
            config["base_url"] = base_url

        try:
            # Let DSPy/LiteLLM handle all model routing and provider logic
            return dspy.LM(**config)
        except Exception as e:
            cls._handle_lm_creation_error(model_name, e)
            raise

    @classmethod
    def _handle_lm_creation_error(cls, model_name: str, error: Exception) -> None:
        """Provide helpful error messages for common LM creation failures."""
        error_str = str(error).lower()

        if "api key" in error_str or "authentication" in error_str:
            # Try to guess which API key might be needed
            if "openai" in error_str or model_name.startswith(("gpt", "o1")):
                suggestion = "Set OPENAI_API_KEY environment variable"
            elif "anthropic" in error_str or "claude" in model_name:
                suggestion = "Set ANTHROPIC_API_KEY environment variable"
            elif "google" in error_str or "gemini" in model_name:
                suggestion = "Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable"
            else:
                suggestion = "Check API key configuration for your provider"

            warnings.warn(f"Authentication error for {model_name}: {suggestion}")

        elif "not found" in error_str or "does not exist" in error_str:
            warnings.warn(
                f"Model {model_name} not found. Check that the model name is correct "
                f"and that you have access to it. See LiteLLM docs for supported models."
            )

        elif "rate limit" in error_str or "quota" in error_str:
            warnings.warn(
                f"Rate limit or quota exceeded for {model_name}. "
                f"Please check your usage limits."
            )

    @classmethod
    def get_default_model(cls) -> str:
        """Auto-detect an available model based on API keys.

        Checks for available API keys and returns a compatible model.

        Returns:
            Model name string for an available model
        """
        # Check for available API keys
        if os.getenv("OPENAI_API_KEY"):
            return "gpt-5-mini"
        elif os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
            return "gemini/gemini-2.5-flash"
        elif os.getenv("ANTHROPIC_API_KEY"):
            return "claude-3-5-haiku-latest"
        elif os.getenv("GROQ_API_KEY"):
            return "groq/llama-3.1-8b-instant"
        elif os.getenv("TOGETHER_API_KEY"):
            return "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"
        else:
            # Check if Ollama is available
            try:
                import subprocess

                result = subprocess.run(
                    ["ollama", "list"], capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0:
                    # Check for available models in Ollama
                    if "llama3.2" in result.stdout:
                        return "ollama/llama3.2"
                    elif "llama3.1" in result.stdout:
                        return "ollama/llama3.1"
                    elif "llama3" in result.stdout:
                        return "ollama/llama3"
                    elif "mistral" in result.stdout:
                        return "ollama/mistral"
                    else:
                        # Default to llama3 which user can pull
                        return "ollama/llama3"
            except (
                subprocess.SubprocessError,
                FileNotFoundError,
                subprocess.TimeoutExpired,
            ):
                pass

            # No API keys found, suggest options
            warnings.warn(
                "No API keys found! Please set one of: OPENAI_API_KEY, "
                "GOOGLE_API_KEY, ANTHROPIC_API_KEY, or install Ollama. "
                "Defaulting to gpt-5-mini which will require an OpenAI API key."
            )
            return "gpt-5-mini"

    @classmethod
    def is_local_model(cls, model_name: str) -> bool:
        """Check if a model runs locally (doesn't need API keys).

        Args:
            model_name: Model name to check

        Returns:
            True if model runs locally, False otherwise
        """
        model_lower = model_name.lower()
        # Check for common local model patterns
        return (
            "ollama/" in model_lower
            or "local/" in model_lower
            or "localhost" in model_lower
            or model_lower.startswith("ollama")
            or model_lower == "mock-model"  # For testing
        )
