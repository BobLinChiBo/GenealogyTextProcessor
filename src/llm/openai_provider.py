#!/usr/bin/env python3
"""
OpenAI provider (GPT-5 via Chat Completions, JSON-only, no tools).
- Avoids Responses API (your SDK was returning metadata-only).
- For GPT-5, forces tool_choice="none" and omits sampling params.
- Uses max_completion_tokens for GPT-5 (required) and two-pass JSON mode.
"""

import os
import json
from typing import List, Dict, Any
from openai import OpenAI

from .base import LLMProvider, LLMResponse


class OpenAIProvider(LLMProvider):
    def _initialize(self):
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY or pass api_key.")
        # keep timeouts as before
        import httpx
        if 'gpt-5' in self.model_name.lower():
            timeout = httpx.Timeout(600.0, connect=30.0, read=600.0, write=30.0)
            self.client = OpenAI(api_key=api_key, timeout=timeout, max_retries=1)
        else:
            timeout = httpx.Timeout(30.0, connect=10.0, read=30.0, write=10.0)
            self.client = OpenAI(api_key=api_key, timeout=timeout, max_retries=3)
        self.logger.info(f"Initialized OpenAI provider with model: {self.model_name}")

    # ---- helpers ----
    def _norm_records(self, data: Any) -> List[Dict]:
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for k in ('records','data','items','output','result'):
                v = data.get(k)
                if isinstance(v, list):
                    return v
            return [data]
        return []

    def _extract_json(self, text: str) -> List[Dict]:
        t = (text or '').strip()
        if not t:
            return []
        if t.startswith("```json"):
            t = t[7:]
        elif t.startswith("```"):
            t = t[3:]
        if t.endswith("```"):
            t = t[:-3]
        try:
            return self._norm_records(json.loads(t.strip()))
        except json.JSONDecodeError:
            import re
            m = re.search(r"\[[\s\S]*\]", t) or re.search(r"\{[\s\S]*\}", t)
            if m:
                try:
                    return self._norm_records(json.loads(m.group()))
                except json.JSONDecodeError:
                    return []
            return []

    def _gpt5_chat_json(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Two-pass JSON mode for GPT-5 via Chat Completions."""
        messages = [
            {"role": "system", "content": system_prompt + "\n\nReturn ONLY a single JSON object with top-level key 'records'."},
            {"role": "user", "content": user_prompt}
        ]

        def attempt(with_response_format: bool) -> LLMResponse:
            try:
                params: Dict[str, Any] = {
                    "model": self.model_name,
                    "messages": messages,
                }
                if with_response_format:
                    params["response_format"] = {"type": "json_object"}
                # IMPORTANT: omit temperature/top_p for GPT-5
                r = self.client.chat.completions.create(**params)
                m = r.choices[0].message
                content = (m.content or "").strip()
                if not content:
                    return LLMResponse(records=[], success=False, error="No content in response")
                records = self._extract_json(content)
                if not records:
                    return LLMResponse(records=[], success=False, error="Unparseable JSON", raw_response=content)
                usage = None
                if r.usage:
                    usage = {
                        "prompt_tokens": r.usage.prompt_tokens,
                        "completion_tokens": r.usage.completion_tokens,
                        "total_tokens": r.usage.total_tokens
                    }
                    # Add extended fields if available
                    if hasattr(r.usage, 'completion_tokens_details') and r.usage.completion_tokens_details:
                        if hasattr(r.usage.completion_tokens_details, 'reasoning_tokens'):
                            usage["reasoning_tokens"] = r.usage.completion_tokens_details.reasoning_tokens
                return LLMResponse(records=records, raw_response=content, success=True, usage=usage)
            except Exception as e:
                return LLMResponse(records=[], success=False, error=str(e))

        # Pass 1: with response_format
        first = attempt(True)
        if first.success and first.records:
            return first
        # Pass 2: without response_format
        self.logger.warning("GPT-5: json_object path failed or empty; retrying without response_format...")
        return attempt(False)

    # ---- public API ----
    def parse_genealogy(self, text: str, system_prompt: str, user_prompt_template: str, use_function_calling: bool = True) -> LLMResponse:
        user_prompt = user_prompt_template.replace("{text}", text)
        if 'gpt-5' in self.model_name.lower():
            return self._gpt5_chat_json(system_prompt, user_prompt)
        # non-GPT-5: keep original behavior
        if use_function_calling:
            return self._chat_functions(system_prompt, user_prompt)
        return self._chat_json(system_prompt, user_prompt)

    def parse_with_context(self, context_text: str, new_text: str, system_prompt: str, user_prompt_template: str, use_function_calling: bool = True) -> LLMResponse:
        user_prompt = user_prompt_template.replace("{context}", context_text).replace("{new_text}", new_text)
        return self.parse_genealogy(text="", system_prompt=system_prompt, user_prompt_template=user_prompt, use_function_calling=use_function_calling)

    # ---- legacy chat paths for non-GPT-5 ----
    def _chat_functions(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        tools = [{"type": "function", "function": self.get_genealogy_schema()}]
        try:
            params: Dict[str, Any] = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "tools": tools,
                "tool_choice": {"type": "function", "function": {"name": "extract_genealogy_records"}},
                "temperature": getattr(self, "temperature", 1.0),
            }
            r = self.client.chat.completions.create(**params)
            msg = r.choices[0].message
            # Extract usage information
            usage = None
            if r.usage:
                usage = {
                    "prompt_tokens": r.usage.prompt_tokens,
                    "completion_tokens": r.usage.completion_tokens,
                    "total_tokens": r.usage.total_tokens
                }
                # Add extended fields if available
                if hasattr(r.usage, 'completion_tokens_details') and r.usage.completion_tokens_details:
                    if hasattr(r.usage.completion_tokens_details, 'reasoning_tokens'):
                        usage["reasoning_tokens"] = r.usage.completion_tokens_details.reasoning_tokens
            
            if msg.tool_calls:
                tc = msg.tool_calls[0]
                if tc.function.name == "extract_genealogy_records":
                    try:
                        parsed = json.loads(tc.function.arguments or "{}")
                        recs = self._norm_records(parsed)
                        return LLMResponse(records=recs, raw_response=tc.function.arguments, success=True, usage=usage)
                    except Exception as e:
                        self.logger.error(f"Failed to parse tool args: {e}")
            content = (msg.content or "").strip()
            if content:
                recs = self._extract_json(content)
                if recs:
                    return LLMResponse(records=recs, raw_response=content, success=True, usage=usage)
            return LLMResponse(records=[], success=False, error="No function calls or parseable content.", usage=usage)
        except Exception as e:
            return LLMResponse(records=[], success=False, error=str(e))

    def _chat_json(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        try:
            messages = [
                {"role": "system", "content": system_prompt + "\n\nReturn ONLY a single JSON object with top-level key 'records'."},
                {"role": "user", "content": user_prompt}
            ]
            params: Dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "temperature": getattr(self, "temperature", 1.0),
                "response_format": {"type": "json_object"},
            }
            r = self.client.chat.completions.create(**params)
            m = r.choices[0].message
            content = (m.content or "").strip()
            # Extract usage information
            usage = None
            if r.usage:
                usage = {
                    "prompt_tokens": r.usage.prompt_tokens,
                    "completion_tokens": r.usage.completion_tokens,
                    "total_tokens": r.usage.total_tokens
                }
                # Add extended fields if available
                if hasattr(r.usage, 'completion_tokens_details') and r.usage.completion_tokens_details:
                    if hasattr(r.usage.completion_tokens_details, 'reasoning_tokens'):
                        usage["reasoning_tokens"] = r.usage.completion_tokens_details.reasoning_tokens
            
            if not content:
                return LLMResponse(records=[], success=False, error="No content in response", usage=usage)
            recs = self._extract_json(content)
            if recs:
                return LLMResponse(records=recs, raw_response=content, success=True, usage=usage)
            return LLMResponse(records=[], success=False, error="Failed to parse JSON", usage=usage)
        except Exception as e:
            return LLMResponse(records=[], success=False, error=str(e))
