#!/usr/bin/env python3
"""
Abstract base class for LLM providers.

Defines the common interface that all LLM providers must implement
for genealogy parsing functionality.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.schema_loader import get_schema_for_function_calling


@dataclass
class LLMResponse:
    """
    Standardized response from LLM providers.
    
    Attributes:
        records: List of extracted genealogy records
        raw_response: Raw response from the LLM (for debugging)
        success: Whether the request was successful
        error: Error message if request failed
        usage: Token usage information (if available)
               Expected fields:
               - prompt_tokens: Number of tokens in the input
               - completion_tokens: Number of tokens in the output
               - total_tokens: Total tokens used (prompt + completion)
               - thoughts_tokens: Thinking/reasoning tokens (Gemini 2.5)
               - reasoning_tokens: Reasoning tokens (OpenAI o1 models)
               - cached_tokens: Tokens read from cache
    """
    records: List[Dict[str, Any]]
    raw_response: Optional[str] = None
    success: bool = True
    error: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All LLM providers must implement this interface to ensure
    consistent behavior across different models.
    """
    
    def __init__(self, 
                 model_name: str,
                 api_key: Optional[str] = None,
                 temperature: float = 0.2,
                 max_retries: int = 3,
                 retry_delay: int = 5,
                 **kwargs):
        """
        Initialize the LLM provider.
        
        Args:
            model_name: Name of the model to use
            api_key: API key for the provider
            temperature: Model temperature (0.0-2.0)
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            **kwargs: Provider-specific configuration
        """
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Store additional provider-specific config
        self.config = kwargs
        
        # Initialize the provider
        self._initialize()
    
    @abstractmethod
    def _initialize(self):
        """
        Initialize the provider-specific client and configuration.
        This method should set up the API client and validate credentials.
        """
        pass
    
    @abstractmethod
    def parse_genealogy(self, 
                        text: str,
                        system_prompt: str,
                        user_prompt_template: str,
                        use_function_calling: bool = True) -> LLMResponse:
        """
        Parse genealogy text using the LLM.
        
        Args:
            text: The genealogy text to parse
            system_prompt: System prompt defining the task
            user_prompt_template: User prompt template with {text} placeholder
            use_function_calling: Whether to use function/tool calling
            
        Returns:
            LLMResponse containing extracted records and metadata
        """
        pass
    
    @abstractmethod
    def parse_with_context(self,
                          context_text: str,
                          new_text: str,
                          system_prompt: str,
                          user_prompt_template: str,
                          use_function_calling: bool = True) -> LLMResponse:
        """
        Parse genealogy text with context from previous chunks.
        
        Args:
            context_text: Context from previous chunk
            new_text: New text to parse
            system_prompt: System prompt defining the task
            user_prompt_template: Template with {context} and {new_text} placeholders
            use_function_calling: Whether to use function/tool calling
            
        Returns:
            LLMResponse containing extracted records and metadata
        """
        pass
    
    def get_genealogy_schema(self) -> Dict[str, Any]:
        """
        Get the function/tool schema for genealogy extraction.
        
        This schema defines the structure of the expected output.
        Can be overridden by providers that need custom schemas.
        
        Returns:
            Dictionary containing the function/tool schema
        """
        return get_schema_for_function_calling()
    
    def validate_response(self, response: LLMResponse) -> bool:
        """
        Validate the LLM response.
        
        Args:
            response: The response to validate
            
        Returns:
            True if response is valid, False otherwise
        """
        if not response.success:
            return False
            
        if not response.records:
            self.logger.warning("Response contains no records")
            return False
            
        # Basic validation of record structure
        required_fields = {'name', 'sex', 'father', 'birth_order', 'courtesy', 
                          'birth_time', 'death_time', 'children', 'info', 'original_text', 'note'}
        
        for i, record in enumerate(response.records):
            missing_fields = required_fields - set(record.keys())
            if missing_fields:
                self.logger.warning(
                    f"Record {i} missing fields: {missing_fields}"
                )
                # Don't fail completely, just log the issue
                
        return True
    
    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about the provider.
        
        Returns:
            Dictionary containing provider information
        """
        return {
            'provider': self.__class__.__name__,
            'model': self.model_name,
            'temperature': self.temperature,
            'config': self.config
        }