"""
LLM Configuration Module
Supports both local (Ollama) and online API-based LLMs
"""

import os
from typing import Optional, Dict, Any
from enum import Enum
import dspy
from pathlib import Path


class LLMProvider(Enum):
    """Supported LLM providers"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    TOGETHER = "together"
    ANYSCALE = "anyscale"


class LLMConfig:
    """
    Centralized LLM configuration manager
    Handles both local (Ollama) and cloud-based API providers
    """

    def __init__(self):
        self.provider = None
        self.model_name = None
        self.lm = None

    @staticmethod
    def _get_api_key(provider: str, env_var: str) -> Optional[str]:
        """
        Safely retrieve API key from environment variables

        Args:
            provider: Name of the provider (for error messages)
            env_var: Environment variable name containing the API key

        Returns:
            API key if found, None otherwise
        """
        api_key = os.environ.get(env_var)
        if not api_key:
            raise ValueError(
                f"API key for {provider} not found. "
                f"Please set the {env_var} environment variable."
            )
        return api_key

    def configure_ollama(
        self,
        model: str = "qwen2.5:7b-instruct-q5_k_m",
        api_base: str = "http://localhost:11434"
    ) -> dspy.LM:
        """
        Configure local Ollama LLM

        Args:
            model: Ollama model name
            api_base: Ollama server URL

        Returns:
            Configured DSPy LM instance
        """
        self.provider = LLMProvider.OLLAMA
        self.model_name = model

        # Ollama doesn't require an API key, but DSPy expects one
        self.lm = dspy.LM(
            f"ollama_chat/{model}",
            api_base=api_base,
            api_key=""
        )

        dspy.settings.configure(lm=self.lm)
        return self.lm

    def configure_openai(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None
    ) -> dspy.LM:
        """
        Configure OpenAI API

        Args:
            model: OpenAI model name (e.g., 'gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo')
            api_key: Optional API key (if not provided, reads from OPENAI_API_KEY env var)

        Returns:
            Configured DSPy LM instance
        """
        self.provider = LLMProvider.OPENAI
        self.model_name = model

        if api_key is None:
            api_key = self._get_api_key("OpenAI", "OPENAI_API_KEY")

        self.lm = dspy.LM(
            f"openai/{model}",
            api_key=api_key
        )

        dspy.settings.configure(lm=self.lm)
        return self.lm

    def configure_anthropic(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None
    ) -> dspy.LM:
        """
        Configure Anthropic Claude API

        Args:
            model: Claude model name (e.g., 'claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022')
            api_key: Optional API key (if not provided, reads from ANTHROPIC_API_KEY env var)

        Returns:
            Configured DSPy LM instance
        """
        self.provider = LLMProvider.ANTHROPIC
        self.model_name = model

        if api_key is None:
            api_key = self._get_api_key("Anthropic", "ANTHROPIC_API_KEY")

        self.lm = dspy.LM(
            f"anthropic/{model}",
            api_key=api_key
        )

        dspy.settings.configure(lm=self.lm)
        return self.lm

    def configure_together(
        self,
        model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        api_key: Optional[str] = None
    ) -> dspy.LM:
        """
        Configure Together AI API

        Args:
            model: Together AI model name
            api_key: Optional API key (if not provided, reads from TOGETHER_API_KEY env var)

        Returns:
            Configured DSPy LM instance
        """
        self.provider = LLMProvider.TOGETHER
        self.model_name = model

        if api_key is None:
            api_key = self._get_api_key("Together AI", "TOGETHER_API_KEY")

        self.lm = dspy.LM(
            f"together_ai/{model}",
            api_key=api_key
        )

        dspy.settings.configure(lm=self.lm)
        return self.lm

    def configure_anyscale(
        self,
        model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        api_key: Optional[str] = None
    ) -> dspy.LM:
        """
        Configure Anyscale Endpoints API

        Args:
            model: Anyscale model name
            api_key: Optional API key (if not provided, reads from ANYSCALE_API_KEY env var)

        Returns:
            Configured DSPy LM instance
        """
        self.provider = LLMProvider.ANYSCALE
        self.model_name = model

        if api_key is None:
            api_key = self._get_api_key("Anyscale", "ANYSCALE_API_KEY")

        self.lm = dspy.LM(
            f"anyscale/{model}",
            api_key=api_key
        )

        dspy.settings.configure(lm=self.lm)
        return self.lm

    def configure_from_env(self) -> dspy.LM:
        """
        Auto-configure LLM based on environment variables

        Environment variables checked (in order):
        1. LLM_PROVIDER: Provider to use (ollama, openai, anthropic, together, anyscale)
        2. LLM_MODEL: Model name for the provider
        3. Provider-specific API key environment variables

        If LLM_PROVIDER is not set, defaults to Ollama

        Returns:
            Configured DSPy LM instance
        """
        provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
        model = os.environ.get("LLM_MODEL")

        if provider == "ollama":
            api_base = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
            default_model = "qwen2.5:7b-instruct-q5_k_m"
            return self.configure_ollama(
                model=model or default_model,
                api_base=api_base
            )

        elif provider == "openai":
            return self.configure_openai(model=model or "gpt-4o-mini")

        elif provider == "anthropic":
            return self.configure_anthropic(model=model or "claude-3-5-sonnet-20241022")

        elif provider == "together":
            return self.configure_together(
                model=model or "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
            )

        elif provider == "anyscale":
            return self.configure_anyscale(
                model=model or "meta-llama/Meta-Llama-3.1-8B-Instruct"
            )

        else:
            raise ValueError(
                f"Unknown LLM provider: {provider}. "
                f"Supported providers: ollama, openai, anthropic, together, anyscale"
            )

    def get_current_config(self) -> Dict[str, Any]:
        """
        Get current LLM configuration info

        Returns:
            Dictionary with provider and model information
        """
        return {
            "provider": self.provider.value if self.provider else None,
            "model": self.model_name,
            "configured": self.lm is not None
        }


# Convenience function for quick setup
def setup_llm(
    provider: str = "ollama",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> dspy.LM:
    """
    Quick setup function for LLM configuration

    Args:
        provider: LLM provider ('ollama', 'openai', 'anthropic', 'together', 'anyscale')
        model: Model name (provider-specific)
        api_key: API key for cloud providers
        **kwargs: Additional provider-specific arguments

    Returns:
        Configured DSPy LM instance

    Examples:
        # Local Ollama
        setup_llm("ollama", "qwen2.5:7b-instruct-q5_k_m")

        # OpenAI
        setup_llm("openai", "gpt-4o-mini", api_key="sk-...")

        # Anthropic Claude
        setup_llm("anthropic", "claude-3-5-sonnet-20241022", api_key="sk-ant-...")

        # From environment variables
        setup_llm()  # Uses LLM_PROVIDER and LLM_MODEL env vars
    """
    config = LLMConfig()

    # If no provider specified, try to configure from environment
    if provider == "ollama" and model is None and api_key is None:
        if os.environ.get("LLM_PROVIDER") or os.environ.get("LLM_MODEL"):
            return config.configure_from_env()

    provider = provider.lower()

    if provider == "ollama":
        return config.configure_ollama(
            model=model or "qwen2.5:7b-instruct-q5_k_m",
            api_base=kwargs.get("api_base", "http://localhost:11434")
        )
    elif provider == "openai":
        return config.configure_openai(model=model or "gpt-4o-mini", api_key=api_key)
    elif provider == "anthropic":
        return config.configure_anthropic(
            model=model or "claude-3-5-sonnet-20241022",
            api_key=api_key
        )
    elif provider == "together":
        return config.configure_together(
            model=model or "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            api_key=api_key
        )
    elif provider == "anyscale":
        return config.configure_anyscale(
            model=model or "meta-llama/Meta-Llama-3.1-8B-Instruct",
            api_key=api_key
        )
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Supported: ollama, openai, anthropic, together, anyscale"
        )
