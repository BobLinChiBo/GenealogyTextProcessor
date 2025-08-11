#!/usr/bin/env python3
"""
LiteLLM provider implementation for genealogy parsing.

This provider uses LiteLLM to provide a unified interface for multiple
LLM providers (OpenAI, Gemini, Anthropic, etc.) with function calling
to extract structured genealogy data.
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional

try:
    import litellm
    from litellm import completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    litellm = None

from .base import LLMProvider, LLMResponse


class LiteLLMProvider(LLMProvider):
    """
    LiteLLM unified provider for genealogy parsing.
    
    Supports multiple providers through a single interface:
    - OpenAI: gpt-3.5-turbo, gpt-4, gpt-4o-mini
    - Google: gemini/gemini-1.5-flash, gemini/gemini-1.5-pro
    - Anthropic: claude-3-opus, claude-3-sonnet
    - And many more...
    
    Uses function calling for structured output when supported.
    """
    
    def _initialize(self):
        """Initialize the LiteLLM provider."""
        if not LITELLM_AVAILABLE:
            raise ImportError(
                "LiteLLM package not installed. "
                "Please install it with: pip install litellm"
            )
        
        # Set up API keys based on the model prefix
        self._setup_api_keys()
        
        # Configure LiteLLM settings
        litellm.drop_params = True  # Drop unsupported params automatically
        litellm.set_verbose = self.config.get('verbose', False)
        
        # Set timeout
        self.timeout = self.config.get('timeout', 120)
        
        self.logger.info(f"Initialized LiteLLM provider with model: {self.model_name}")
    
    def _setup_api_keys(self):
        """Set up API keys based on the model being used."""
        model_lower = self.model_name.lower()
        
        # Determine which API key to use based on model
        if model_lower.startswith('gpt') or model_lower.startswith('openai'):
            api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
        elif 'gemini' in model_lower:
            api_key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if api_key:
                os.environ["GEMINI_API_KEY"] = api_key
        elif 'claude' in model_lower:
            api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                os.environ["ANTHROPIC_API_KEY"] = api_key
        
        # Allow custom API base for local models or proxies
        if self.config.get('api_base'):
            os.environ["OPENAI_API_BASE"] = self.config['api_base']
    
    def parse_genealogy(self, 
                        text: str,
                        system_prompt: str,
                        user_prompt_template: str,
                        use_function_calling: bool = True) -> LLMResponse:
        """
        Parse genealogy text using LiteLLM.
        
        Args:
            text: The genealogy text to parse
            system_prompt: System prompt defining the task
            user_prompt_template: User prompt template with {text} placeholder
            use_function_calling: Whether to use function calling
            
        Returns:
            LLMResponse containing extracted records
        """
        user_prompt = user_prompt_template.replace("{text}", text)
        
        for attempt in range(self.max_retries):
            try:
                # Check if model supports function calling
                supports_functions = self._supports_function_calling()
                
                if use_function_calling and supports_functions:
                    response = self._call_with_functions(system_prompt, user_prompt)
                else:
                    response = self._call_json_mode(system_prompt, user_prompt)
                
                if response.success:
                    return response
                    
            except Exception as e:
                self.logger.error(f"API error on attempt {attempt + 1}: {e}")
                
                # Check for quota errors
                if any(err in str(e).lower() for err in ['quota', '429', 'rate_limit']):
                    return LLMResponse(
                        records=[],
                        success=False,
                        error=f"API quota/rate limit exceeded: {e}"
                    )
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    return LLMResponse(
                        records=[],
                        success=False,
                        error=f"Failed after {self.max_retries} attempts: {e}"
                    )
        
        return LLMResponse(records=[], success=False, error="Max retries exceeded")
    
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
            use_function_calling: Whether to use function calling
            
        Returns:
            LLMResponse containing extracted records
        """
        user_prompt = user_prompt_template.replace(
            "{context}", context_text
        ).replace(
            "{new_text}", new_text
        )
        
        return self.parse_genealogy(
            text="",  # Not used since we have custom prompt
            system_prompt=system_prompt,
            user_prompt_template=user_prompt,
            use_function_calling=use_function_calling
        )
    
    def _supports_function_calling(self) -> bool:
        """
        Check if the current model supports function calling.
        
        Returns:
            True if model supports function calling
        """
        model_lower = self.model_name.lower()
        
        # Models that support function calling
        function_models = [
            'gpt-3.5-turbo', 'gpt-4', 'gpt-4o',
            'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-2',
            'claude-3', 'claude-instant',
            'mistral-large', 'mistral-medium'
        ]
        
        return any(fm in model_lower for fm in function_models)
    
    def _call_with_functions(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """
        Call LiteLLM with function calling.
        
        Args:
            system_prompt: System message
            user_prompt: User message
            
        Returns:
            LLMResponse with extracted records
        """
        # Convert schema to LiteLLM tools format
        tools = [{
            "type": "function",
            "function": self.get_genealogy_schema()
        }]
        
        try:
            response = completion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                tools=tools,
                tool_choice="auto",  # Let LiteLLM handle the tool choice
                temperature=self.temperature,
                timeout=self.timeout
            )
            
            # Extract function call response
            message = response.choices[0].message
            
            # Get usage info with extended fields
            usage = None
            if hasattr(response, 'usage'):
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
                
                # Add extended fields if available
                if hasattr(response.usage, 'completion_tokens_details'):
                    details = response.usage.completion_tokens_details
                    if details and hasattr(details, 'reasoning_tokens'):
                        usage["reasoning_tokens"] = details.reasoning_tokens
                    if details and hasattr(details, 'thoughts_tokens'):
                        usage["thoughts_tokens"] = details.thoughts_tokens
                
                # Check for cached tokens
                if hasattr(response.usage, 'prompt_tokens_details'):
                    details = response.usage.prompt_tokens_details
                    if details and hasattr(details, 'cached_tokens'):
                        usage["cached_tokens"] = details.cached_tokens
                
                # LiteLLM may expose provider-specific fields
                if hasattr(response.usage, 'cache_read_input_tokens'):
                    usage["cached_tokens"] = response.usage.cache_read_input_tokens
            
            # Check for tool calls
            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_call = message.tool_calls[0]
                if tool_call.function.name == "extract_genealogy_records":
                    try:
                        result = json.loads(tool_call.function.arguments)
                        records = result.get("records", [])
                        
                        self.logger.info(f"Successfully extracted {len(records)} records via function calling")
                        
                        return LLMResponse(
                            records=records,
                            raw_response=tool_call.function.arguments,
                            success=True,
                            usage=usage
                        )
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Failed to parse function response: {e}")
                        return LLMResponse(
                            records=[],
                            raw_response=tool_call.function.arguments,
                            success=False,
                            error=f"JSON decode error: {e}",
                            usage=usage
                        )
            
            # If no tool calls, try to extract from content
            if hasattr(message, 'content') and message.content:
                records = self._extract_json_from_response(message.content)
                if records:
                    return LLMResponse(
                        records=records,
                        raw_response=message.content,
                        success=True,
                        usage=usage
                    )
            
            return LLMResponse(
                records=[],
                success=False,
                error="No function calls or valid JSON in response",
                usage=usage
            )
            
        except Exception as e:
            self.logger.error(f"Function calling failed: {e}")
            raise
    
    def _call_json_mode(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """
        Call LiteLLM with JSON response format (fallback mode).
        
        Args:
            system_prompt: System message
            user_prompt: User message
            
        Returns:
            LLMResponse with extracted records
        """
        try:
            # Add JSON instruction to prompts
            json_system = f"{system_prompt}\n\nYou must respond with valid JSON only."
            json_user = f"{user_prompt}\n\nIMPORTANT: Output a JSON array directly at root level: [{{...}}, {{...}}]"
            
            # Some models support response_format
            kwargs = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": json_system},
                    {"role": "user", "content": json_user}
                ],
                "temperature": self.temperature,
                "timeout": self.timeout
            }
            
            # Add response_format for models that support it (OpenAI)
            if 'gpt' in self.model_name.lower():
                kwargs["response_format"] = {"type": "json_object"}
            
            response = completion(**kwargs)
            
            response_text = response.choices[0].message.content
            
            # Get usage info with extended fields
            usage = None
            if hasattr(response, 'usage'):
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
                
                # Add extended fields if available
                if hasattr(response.usage, 'completion_tokens_details'):
                    details = response.usage.completion_tokens_details
                    if details and hasattr(details, 'reasoning_tokens'):
                        usage["reasoning_tokens"] = details.reasoning_tokens
                    if details and hasattr(details, 'thoughts_tokens'):
                        usage["thoughts_tokens"] = details.thoughts_tokens
                
                # Check for cached tokens
                if hasattr(response.usage, 'prompt_tokens_details'):
                    details = response.usage.prompt_tokens_details
                    if details and hasattr(details, 'cached_tokens'):
                        usage["cached_tokens"] = details.cached_tokens
                
                # LiteLLM may expose provider-specific fields
                if hasattr(response.usage, 'cache_read_input_tokens'):
                    usage["cached_tokens"] = response.usage.cache_read_input_tokens
            
            # Parse JSON response
            records = self._extract_json_from_response(response_text)
            
            return LLMResponse(
                records=records,
                raw_response=response_text,
                success=len(records) > 0,
                usage=usage
            )
            
        except Exception as e:
            self.logger.error(f"JSON mode failed: {e}")
            raise
    
    def _extract_json_from_response(self, response_text: str) -> List[Dict]:
        """
        Extract JSON array from model's response text.
        
        Args:
            response_text: Raw response text from model
            
        Returns:
            List of record dictionaries
        """
        if not response_text:
            return []
            
        text = response_text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        try:
            data = json.loads(text.strip())
            
            # Handle different response formats
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # Check for common wrapper patterns
                for key in ['data', 'output', 'records', 'items', 'results']:
                    if key in data and isinstance(data[key], list):
                        return data[key]
                # Single object, wrap in list
                return [data]
            else:
                self.logger.warning(f"Unexpected response type: {type(data)}")
                return []
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            self.logger.debug(f"Response: {text[:500]}...")
            return []
    
    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about the provider.
        
        Returns:
            Dictionary containing provider information
        """
        info = super().get_provider_info()
        info['supports_function_calling'] = self._supports_function_calling()
        info['litellm_version'] = litellm.__version__ if litellm else 'Not installed'
        return info