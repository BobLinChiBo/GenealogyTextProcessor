#!/usr/bin/env python3
"""
LLM providers for genealogy parsing.

This module provides a unified interface for different LLM providers
(OpenAI, Google Gemini, etc.) to extract structured genealogy data.
"""

from .base import LLMProvider, LLMResponse
from .openai_provider import OpenAIProvider
from .gemini_provider import GeminiProvider
from .litellm_provider import LiteLLMProvider

__all__ = [
    'LLMProvider',
    'LLMResponse',
    'OpenAIProvider',
    'GeminiProvider',
    'LiteLLMProvider',
    'get_provider'
]


def get_provider(provider_name: str, **kwargs) -> LLMProvider:
    """
    Factory function to create LLM provider instances.
    
    Args:
        provider_name: Name of the provider ('openai', 'gemini', 'litellm')
        **kwargs: Provider-specific configuration
        
    Returns:
        LLMProvider instance
        
    Raises:
        ValueError: If provider name is not recognized
    """
    providers = {
        'openai': OpenAIProvider,
        'gemini': GeminiProvider,
        'litellm': LiteLLMProvider
    }
    
    provider_class = providers.get(provider_name.lower())
    if not provider_class:
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            f"Available providers: {', '.join(providers.keys())}"
        )
    
    return provider_class(**kwargs)