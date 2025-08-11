#!/usr/bin/env python3
"""
Google Gemini provider implementation for genealogy parsing.

This provider uses the Google Generative AI API (Gemini) with function calling
to extract structured genealogy data.
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
from collections.abc import Mapping, Sequence

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

from .base import LLMProvider, LLMResponse


class GeminiProvider(LLMProvider):
    """
    Google Gemini provider for genealogy parsing.
    
    Supports models like gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash.
    Uses function calling for structured output.
    """
    
    def _initialize(self):
        """Initialize the Gemini client."""
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Google Generative AI package not installed. "
                "Please install it with: pip install google-generativeai"
            )
        
        # Get API key from parameter or environment
        api_key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key not found. Please provide it or set "
                "the GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
            )
        
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Initialize the model with safety settings
        # Try more permissive settings for genealogy content
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        # Add CIVIC_INTEGRITY if available (Gemini 2.5 models)
        try:
            if hasattr(HarmCategory, 'HARM_CATEGORY_CIVIC_INTEGRITY'):
                self.safety_settings[HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY] = HarmBlockThreshold.BLOCK_NONE
        except:
            pass
        
        # Create generation config
        self.generation_config = genai.GenerationConfig(
            temperature=self.temperature,
            response_mime_type="application/json" if not self.config.get('use_function_calling', True) else None
        )
        
        self.logger.info(f"Initialized Gemini provider with model: {self.model_name}")
    
    def parse_genealogy(self, 
                        text: str,
                        system_prompt: str,
                        user_prompt_template: str,
                        use_function_calling: bool = True) -> LLMResponse:
        """
        Parse genealogy text using Google Gemini.
        
        Args:
            text: The genealogy text to parse
            system_prompt: System prompt defining the task
            user_prompt_template: User prompt template with {text} placeholder
            use_function_calling: Whether to use function calling
            
        Returns:
            LLMResponse containing extracted records
        """
        user_prompt = user_prompt_template.replace("{text}", text)
        
        # Combine system and user prompts for Gemini
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        for attempt in range(self.max_retries):
            try:
                if use_function_calling:
                    response = self._call_with_functions(full_prompt)
                else:
                    response = self._call_json_mode(full_prompt)
                
                if response.success:
                    return response
                    
            except Exception as e:
                self.logger.error(f"API error on attempt {attempt + 1}: {e}")
                
                # Check for quota errors
                if "quota" in str(e).lower() or "429" in str(e):
                    return LLMResponse(
                        records=[],
                        success=False,
                        error=f"API quota exceeded: {e}"
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
        # Fill in the context and new_text in the template
        user_prompt = user_prompt_template.replace(
            "{context}", context_text
        ).replace(
            "{new_text}", new_text
        )
        
        # Combine system and user prompts for Gemini
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Call the API directly with the complete prompt
        for attempt in range(self.max_retries):
            try:
                if use_function_calling:
                    response = self._call_with_functions(full_prompt)
                else:
                    # Use JSON mode
                    response = self._call_json_mode(full_prompt)
                
                if response.success or not response.error:
                    return response
                    
            except Exception as e:
                self.logger.error(f"API error on attempt {attempt + 1}: {e}")
                
                # Check for quota errors
                if "quota" in str(e).lower() or "429" in str(e):
                    wait_time = self.retry_delay * (attempt + 1)
                    self.logger.info(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                    
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return LLMResponse(
                        records=[],
                        success=False,
                        error=f"Failed after {self.max_retries} attempts: {str(e)}"
                    )
        
        return LLMResponse(
            records=[],
            success=False,
            error="Failed to get response from Gemini API"
        )
    
    def _call_with_python_function(self, prompt: str) -> LLMResponse:
        """
        Call Gemini API with simplified Python function approach.
        Uses automatic function calling with Python functions.
        
        Args:
            prompt: Combined system and user prompt
            
        Returns:
            LLMResponse with extracted records
        """
        try:
            # Define a Python function for genealogy extraction
            def extract_genealogy_records(records: list[dict]) -> dict:
                """
                Extract structured genealogy records from Chinese family tree text.
                      
                Returns:
                    Dictionary containing the array of records
                """
                return {"records": records}
            
            # Initialize model with the Python function as a tool
            model = genai.GenerativeModel(
                model_name=self.model_name,
                tools=[extract_genealogy_records],
                safety_settings=self.safety_settings,
                generation_config=self.generation_config
            )
            
            # Start a chat session with automatic function calling
            chat = model.start_chat(enable_automatic_function_calling=True)
            
            # Send the prompt
            response = chat.send_message(prompt)
            
            # The response should contain the extracted records
            # With automatic function calling, the model will call the function
            # and return the result in the response
            
            # Check if we got a valid response
            if hasattr(response, 'text'):
                try:
                    # Try to parse the response as JSON
                    records = self._extract_json_from_response(response.text)
                    if records:
                        return LLMResponse(
                            records=records,
                            raw_response=response.text,
                            success=True,
                            usage=self._extract_usage(response)
                        )
                except Exception as e:
                    self.logger.debug(f"Could not parse response as JSON: {e}")
            
            # Check chat history for function call results
            for message in chat.history:
                if hasattr(message, 'parts'):
                    for part in message.parts:
                        if hasattr(part, 'function_response') and part.function_response:
                            # Extract records from function response
                            response_dict = type(part.function_response.response).to_dict(
                                part.function_response.response
                            )
                            records = response_dict.get("records", [])
                            if records:
                                self.logger.info(f"Extracted {len(records)} records via Python function")
                                return LLMResponse(
                                    records=records,
                                    raw_response=str(part.function_response.response),
                                    success=True,
                                    usage=self._extract_usage(response)
                                )
            
            # If we still don't have records, fall back to JSON mode
            self.logger.warning("Python function approach didn't yield results, falling back to JSON mode")
            return self._call_json_mode(prompt)
            
        except Exception as e:
            self.logger.error(f"Python function calling failed: {e}")
            # Fall back to JSON mode
            return self._call_json_mode(prompt)

    def _json_schema_to_protos(self, schema: Dict[str, Any]) -> "genai.protos.Schema":
        """
        Convert a JSON-schema-like dict (from base.get_genealogy_schema) to Gemini protos.Schema.
        Notes:
        - Gemini protos do NOT support NULL/ONE_OF; we drop 'null' from unions.
        - 'required' is not enforced at the proto level; keep validation in code.
        - additionalProperties/strict are ignored (not supported in protos).
        """
        t = schema.get("type")

        # If type is a union (e.g. ["string", "null"]), pick the first non-null type.
        if isinstance(t, list):
            non_null = [tt for tt in t if tt != "null"]
            t = non_null[0] if non_null else "string"

        # Default unknown to string
        if t is None:
            t = "string"

        if t == "object":
            props = {}
            for name, subschema in schema.get("properties", {}).items():
                props[name] = self._json_schema_to_protos(subschema)
            # Gemini protos have no explicit "required" handling
            return genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties=props,
            )

        if t == "array":
            items = schema.get("items") or {"type": "string"}
            return genai.protos.Schema(
                type=genai.protos.Type.ARRAY,
                items=self._json_schema_to_protos(items),
            )

        if t == "string":
            return genai.protos.Schema(type=genai.protos.Type.STRING)
        if t == "integer":
            return genai.protos.Schema(type=genai.protos.Type.INTEGER)
        if t == "number":
            return genai.protos.Schema(type=genai.protos.Type.NUMBER)
        if t == "boolean":
            return genai.protos.Schema(type=genai.protos.Type.BOOLEAN)

        # Fallback
        return genai.protos.Schema(type=genai.protos.Type.STRING)

    def _to_python_dict(self, obj):
        """Convert Gemini SDK objects (MapComposite/RepeatedComposite/protobuf) to plain Python types."""
        # Primitives pass through
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj

        # Already dict
        if isinstance(obj, dict):
            return {k: self._to_python_dict(v) for k, v in obj.items()}

        # Generic Mapping (e.g., MapComposite)
        if isinstance(obj, Mapping):
            try:
                return {k: self._to_python_dict(v) for k, v in dict(obj).items()}
            except Exception:
                # Fallback: iterate items if provided
                try:
                    return {k: self._to_python_dict(v) for k, v in obj.items()}  # type: ignore
                except Exception:
                    return str(obj)

        # Generic Sequence BUT not bytes/str
        if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
            try:
                return [self._to_python_dict(x) for x in list(obj)]
            except Exception:
                try:
                    return [self._to_python_dict(x) for x in obj]  # type: ignore
                except Exception:
                    return str(obj)

        # Protobuf message/struct → dict
        try:
            from google.protobuf.json_format import MessageToDict
            return MessageToDict(obj, preserving_proto_field_name=True)
        except Exception:
            pass

        # Last resort: string
        return str(obj)


    def _normalize_records(self, recs):
        """Ensure records -> list[dict] of plain JSON-serializable structures."""
        recs = self._to_python_dict(recs)
        if recs is None:
            return []

        # If it's not a list after conversion, wrap it
        if not isinstance(recs, list):
            recs = [recs]

        out = []
        for r in recs:
            r = self._to_python_dict(r)
            if isinstance(r, dict):
                out.append(r)
            else:
                out.append({"value": r})
        return out


    def _json_default(self, o):
        """Safe default for json.dumps to catch any remaining SDK objects."""
        converted = self._to_python_dict(o)
        # After conversion, if still not JSON-serializable, stringify
        try:
            json.dumps(converted)
            return converted
        except Exception:
            return str(converted)

    def _strip_schema_descriptions(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Return a copy of schema with 'description' fields removed to reduce payload size."""
        def _strip(node):
            if isinstance(node, dict):
                node = {k: _strip(v) for k, v in node.items() if k != "description"}
            elif isinstance(node, list):
                node = [_strip(v) for v in node]
            return node
        return _strip(json.loads(json.dumps(schema)))


    def _call_with_functions(self, prompt: str) -> LLMResponse:
        """
        Call Gemini with function calling using the single schema from base.py.
        Retries on 5xx, uses AUTO tool mode, normalizes outputs to plain JSON.
        """
        import json
        import time

        try:
            # 1) Single source of truth for schema (from base.py), trim descriptions to reduce payload
            base_schema = self.get_genealogy_schema()
            try:
                base_schema = self._strip_schema_descriptions(base_schema)
            except Exception:
                # helper is optional; continue if not present
                pass

            fn_name = base_schema.get("name", "extract_genealogy_records")
            fn_desc = base_schema.get("description", "Extract structured genealogy records from Chinese family tree text")
            params = base_schema.get("parameters", {"type": "object", "properties": {}})

            # 2) Convert JSON-like schema -> Gemini protos.Schema
            protos_schema = self._json_schema_to_protos(params)

            # 3) Build tool
            extract_genealogy = genai.protos.FunctionDeclaration(
                name=fn_name,
                description=fn_desc,
                parameters=protos_schema
            )
            tool = genai.protos.Tool(function_declarations=[extract_genealogy])

            # 4) Init model
            model = genai.GenerativeModel(
                model_name=self.model_name,
                tools=[tool],
                safety_settings=self.safety_settings,
                generation_config=self.generation_config
            )

            # 5) Generate with retries on 5xx (fallback to JSON mode if still failing)
            last_err = None
            response = None
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = model.generate_content(
                        prompt,
                        tool_config={"function_calling_config": {"mode": "AUTO"}},
                        request_options={"timeout": 120},
                    )
                    last_err = None
                    break
                except Exception as e:
                    msg = str(e)
                    last_err = e
                    if "500" in msg or " 5" in msg or "internal error" in msg.lower():
                        sleep_s = min(8, self.retry_delay * attempt) + (0.2 * attempt)
                        self.logger.warning(f"5xx from Gemini (attempt {attempt}/{self.max_retries}). Sleeping {sleep_s:.1f}s …")
                        time.sleep(sleep_s)
                        continue
                    raise  # non-5xx

            if last_err is not None:
                self.logger.error(f"Gemini tool call failed after retries: {last_err}")
                return self._call_json_mode(prompt)

            # 6) Basic checks
            if not getattr(response, "candidates", None):
                self.logger.error("No candidates in response - likely blocked by safety filters")
                if hasattr(response, "prompt_feedback"):
                    self.logger.error(f"Prompt feedback: {response.prompt_feedback}")
                return LLMResponse(
                    records=[],
                    success=False,
                    error="Response blocked by safety filters",
                    usage=self._extract_usage(response),
                )

            cand = response.candidates[0]
            fr = getattr(cand, "finish_reason", None)
            # 1=STOP, 2=MAX_TOKENS, 3=SAFETY, 4=RECITATION, 5=OTHER, 10=MALFORMED_FUNCTION_CALL
            if fr == 2 or (isinstance(fr, str) and fr.endswith("MAX_TOKENS")):
                self.logger.warning("Response truncated due to max tokens")
            elif fr == 3 or (isinstance(fr, str) and fr.endswith("SAFETY")):
                self.logger.error(f"Response blocked by safety filters. Finish reason: {fr}")
                if hasattr(cand, "safety_ratings"):
                    self.logger.error(f"Safety ratings: {cand.safety_ratings}")
                return LLMResponse(
                    records=[],
                    success=False,
                    error="Content blocked by safety filters",
                    usage=self._extract_usage(response),
                )
            elif fr == 10 or (isinstance(fr, str) and fr.endswith("MALFORMED_FUNCTION_CALL")):
                self.logger.error(f"Malformed function call. Finish reason: {fr}")
                return self._call_json_mode(prompt)
            elif fr not in [1, None] and not (isinstance(fr, str) and fr.endswith("STOP")):
                self.logger.warning(f"Unexpected finish reason: {fr}")

            # 7) Preferred: extract function_call args from parts
            if hasattr(cand, "content") and hasattr(cand.content, "parts"):
                for part in cand.content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        if fc.name == fn_name:
                            args_dict = self._to_python_dict(fc.args) or {}
                            records = self._normalize_records(args_dict.get("records", []))
                            return LLMResponse(
                                records=records,
                                raw_response=json.dumps(args_dict, ensure_ascii=False, default=self._json_default),
                                success=True,
                                usage=self._extract_usage(response),
                            )

            # 8) Legacy attribute path
            if hasattr(cand, "function_calls") and cand.function_calls:
                fc = cand.function_calls[0]
                if fc.name == fn_name:
                    args_dict = self._to_python_dict(fc.args) or {}
                    records = self._normalize_records(args_dict.get("records", []))
                    return LLMResponse(
                        records=records,
                        raw_response=json.dumps(args_dict, ensure_ascii=False, default=self._json_default),
                        success=True,
                        usage=self._extract_usage(response),
                    )

            # 9) Fallback: try text → JSON (prefer concatenated text parts)
            try:
                text_buf = []
                if hasattr(cand, "content") and hasattr(cand.content, "parts"):
                    for p in cand.content.parts:
                        if hasattr(p, "text") and p.text:
                            text_buf.append(p.text)
                text_response = "".join(text_buf).strip() if text_buf else (getattr(response, "text", "") or "")
                if text_response:
                    records = self._extract_json_from_response(text_response)
                    records = self._normalize_records(records)
                    if records:
                        return LLMResponse(
                            records=records,
                            raw_response=text_response,
                            success=True,
                            usage=self._extract_usage(response),
                        )
            except Exception as e:
                self.logger.debug(f"Text fallback failed: {e}")

            # 10) Nothing usable
            return LLMResponse(
                records=[],
                success=False,
                error="No function calls or valid JSON in response",
                usage=self._extract_usage(response),
            )

        except Exception as e:
            self.logger.error(f"Function calling failed: {e}")
            raise

    def _call_json_mode(self, prompt: str) -> LLMResponse:
        """
        Call Gemini API with JSON response format (fallback mode).
        
        Args:
            prompt: Combined system and user prompt
            
        Returns:
            LLMResponse with extracted records
        """
        try:
            # Initialize model without tools for JSON mode
            model = genai.GenerativeModel(
                model_name=self.model_name,
                safety_settings=self.safety_settings,
                generation_config=genai.GenerationConfig(
                    temperature=self.temperature,
                    response_mime_type="application/json"
                )
            )
            
            # Add JSON instruction to prompt
            json_prompt = f"{prompt}\n\nIMPORTANT: Output a JSON array directly at root level: [{{...}}, {{...}}]"
            
            # Generate response
            response = model.generate_content(json_prompt)
            
            # Check if response was blocked
            if not response.candidates:
                self.logger.error("No candidates in response - likely blocked by safety filters")
                return LLMResponse(
                    records=[],
                    success=False,
                    error="Response blocked by safety filters",
                    usage=self._extract_usage(response)
                )
            
            # Check finish reason
            candidate = response.candidates[0]
            if hasattr(candidate, 'finish_reason'):
                finish_reason = candidate.finish_reason
                if finish_reason == 2 or str(finish_reason).endswith('MAX_TOKENS'):
                    self.logger.warning("Response truncated due to max tokens")
                elif finish_reason == 3 or str(finish_reason).endswith('SAFETY'):
                    self.logger.error(f"Response blocked by safety. Finish reason: {finish_reason}")
                    return LLMResponse(
                        records=[],
                        success=False,
                        error="Content blocked by safety filters",
                        usage=self._extract_usage(response)
                    )
                elif finish_reason not in [1, None] and not str(finish_reason).endswith('STOP'):
                    self.logger.warning(f"Unexpected finish reason: {finish_reason}")
            
            if hasattr(response, 'text') and response.text:
                records = self._extract_json_from_response(response.text)
                
                return LLMResponse(
                    records=records,
                    raw_response=response.text,
                    success=len(records) > 0,
                    usage=self._extract_usage(response)
                )
            
            return LLMResponse(
                records=[],
                success=False,
                error="No text in response",
                usage=self._extract_usage(response)
            )
            
        except Exception as e:
            self.logger.error(f"JSON mode failed: {e}")
            # Check if this is a token limit issue
            if "finish_reason" in str(e) and "is 2" in str(e):
                print(f"\n⚠️ Chunk too large for Gemini. Reduce chunk_size in config.yaml")
                print(f"   Recommended: chunk_size: 3-5 lines for complex genealogy text")
            raise
    
    def _extract_json_from_response(self, response_text: str) -> List[Dict]:
        """
        Extract JSON array from model's response text.
        
        Args:
            response_text: Raw response text from model
            
        Returns:
            List of record dictionaries
        """
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
                for key in ['data', 'output', 'records', 'items']:
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
    
    def _extract_usage(self, response) -> Optional[Dict[str, int]]:
        """
        Extract token usage information from Gemini response.
        
        Args:
            response: Gemini API response
            
        Returns:
            Dictionary with token usage or None
        """
        try:
            if hasattr(response, 'usage_metadata'):
                usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count
                }
                
                # Add thoughts_token_count if available (Gemini 2.5 models with thinking)
                if hasattr(response.usage_metadata, 'thoughts_token_count'):
                    usage["thoughts_tokens"] = response.usage_metadata.thoughts_token_count
                elif hasattr(response.usage_metadata, 'thoughtsTokenCount'):
                    usage["thoughts_tokens"] = response.usage_metadata.thoughtsTokenCount
                
                # Check for cached tokens
                if hasattr(response.usage_metadata, 'cached_content_token_count'):
                    usage["cached_tokens"] = response.usage_metadata.cached_content_token_count
                    
                return usage
        except Exception as e:
            self.logger.debug(f"Failed to extract usage metadata: {e}")
        return None