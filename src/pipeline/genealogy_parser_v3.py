#!/usr/bin/env python3
"""
genealogy_parser_v3.py

Uses the modular LLM provider system to parse cleaned genealogy text files,
extracting structured information about family relationships. This version
supports multiple LLM providers (OpenAI, Gemini, LiteLLM) through a unified interface.

Includes checkpoint/resume functionality for handling API quota limits.
"""

import os
import sys
import json
import logging
import time
import argparse
import signal
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import LLM providers
from llm import get_provider, LLMProvider, LLMResponse

# Import centralized prompts
from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, USER_PROMPT_WITH_CONTEXT

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Try to load .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed

# Optional: Pydantic for validation
try:
    from pydantic import BaseModel, Field, field_validator, ConfigDict
    from typing import Optional
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object  # Dummy for when Pydantic isn't available
    Field = lambda *args, **kwargs: None

# --- Data Structures ---
# Aligned with the desired JSON output for clarity and type safety.

if PYDANTIC_AVAILABLE:
    class ChildRecord(BaseModel):
        """Structured record for a child, including birth order."""
        model_config = ConfigDict(extra='forbid')
        
        order: Optional[int] = Field(default=None, ge=1, description="Birth order, must be positive if provided")
        name: Optional[str] = Field(default=None, description="Child's name")
        sex: Optional[str] = Field(default=None, description="Sex of the child (male/female/null)")
        
        @field_validator('name')
        @classmethod
        def validate_name(cls, v: Optional[str]) -> Optional[str]:
            """Ensure name is not just whitespace if provided."""
            if v is None:
                return None
            if not v.strip():
                return None
            return v.strip()
        
        @field_validator('sex')
        @classmethod
        def validate_sex(cls, v: Optional[str]) -> Optional[str]:
            """Validate sex value."""
            if v is None or v == "null":
                return None
            if v not in ["male", "female"]:
                return None
            return v
    
    class PersonRecord(BaseModel):
        """Structured record for a person in the genealogy."""
        model_config = ConfigDict(extra='forbid')
        
        name: Optional[str] = Field(default=None, description="Person's name")
        sex: Optional[str] = Field(default=None, description="Sex of the person (male/female/null)")
        father: Optional[str] = Field(default=None, description="Father's name")
        birth_order: Optional[int] = Field(default=None, ge=1, description="Birth order number (1 for 長子, 2 for 次子, etc.)")
        courtesy: Optional[str] = Field(default=None, description="Courtesy name (字)")
        birth_time: Optional[str] = Field(default=None, description="Birth time information")
        death_time: Optional[str] = Field(default=None, description="Death time information")
        children: List[ChildRecord] = Field(default_factory=list, description="List of children")
        info: Optional[str] = Field(default=None, description="Biographical information")
        original_text: str = Field(default="", description="Original text line")
        note: Optional[str] = Field(default=None, description="Processing notes")
        is_update_for_previous: Optional[bool] = Field(default=None, description="Update flag for merging")
        skip: bool = Field(default=False, description="Skip this record if it's noise or invalid")
        
        @field_validator('children', mode='before')
        @classmethod
        def validate_children(cls, v):
            """Ensure children is a list and convert dicts to ChildRecord."""
            if v is None:
                return []
            if not isinstance(v, list):
                return []
            result = []
            for child in v:
                if isinstance(child, dict):
                    result.append(ChildRecord(**child))
                elif isinstance(child, ChildRecord):
                    result.append(child)
            return result

else:
    # Fallback to dataclasses when Pydantic is not available
    @dataclass
    class ChildRecord:
        """Structured record for a child, including birth order."""
        order: Optional[int]
        name: Optional[str]
        sex: Optional[str]

    @dataclass
    class PersonRecord:
        """Structured record for a person in the genealogy."""
        name: Optional[str]
        sex: Optional[str]
        father: Optional[str]
        birth_order: Optional[int]
        courtesy: Optional[str]
        birth_time: Optional[str] = None
        death_time: Optional[str] = None
        children: List[ChildRecord] = field(default_factory=list)
        info: Optional[str] = None
        original_text: str = ""
        note: Optional[str] = None
        is_update_for_previous: Optional[bool] = None
        skip: bool = False


# --- Main Parser Class ---

class GenealogyParser:
    """Handles AI-powered parsing of genealogy text using modular LLM providers."""

    # Prompts are now imported from centralized module at the top of the file

    def __init__(self,
                 input_file: str,
                 output_file: str = "genealogy_data.json",
                 provider: str = "litellm",
                 model_name: str = "gpt-4o-mini",
                 api_key: Optional[str] = None,
                 max_retries: int = 3,
                 retry_delay: int = 5,
                 encoding: str = "utf-8",
                 temperature: float = 0.2,
                 use_function_calling: bool = True,
                 checkpoint_dir: Optional[str] = None,
                 resume: bool = False,
                 save_intermediate: bool = True,
                 merge_updates: bool = True,
                 **provider_config):
        """
        Initialize the genealogy parser with LLM provider.
        
        Args:
            input_file: Path to input text file
            output_file: Path to output JSON file
            provider: LLM provider name ('openai', 'gemini', 'litellm')
            model_name: Model name (provider-specific)
            api_key: API key for the provider
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries
            encoding: File encoding
            temperature: Model temperature
            use_function_calling: Whether to use function calling
            checkpoint_dir: Directory for checkpoints
            resume: Whether to resume from checkpoint
            save_intermediate: Whether to save intermediate results
            merge_updates: Whether to merge records with is_update_for_previous flag
            **provider_config: Additional provider-specific configuration
        """
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.encoding = encoding
        self.use_function_calling = use_function_calling
        self.resume = resume
        self.save_intermediate = save_intermediate
        self.merge_updates = merge_updates
        
        # Setup checkpoint directory
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
        else:
            self.checkpoint_dir = self.output_file.parent / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint file paths - Use deterministic hash for consistent checkpoints
        input_hash = hashlib.md5(str(self.input_file).encode()).hexdigest()[:8]
        self.checkpoint_file = self.checkpoint_dir / f"checkpoint_{input_hash}.json"
        self.records_checkpoint_file = self.checkpoint_dir / f"records_{input_hash}.json"
        
        # Initialize LLM provider
        self.llm_provider = self._setup_provider(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_retries=max_retries,
            retry_delay=retry_delay,
            **provider_config
        )
        self.logger = logging.getLogger("genealogy_parser")
        self.stats = {
            "lines_processed": 0, 
            "records_created": 0, 
            "api_calls": 0, 
            "errors": [],
            "total_tokens": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "thoughts_tokens": 0,
                "reasoning_tokens": 0,
                "cached_tokens": 0
            }
        }
        self._interrupted = False
        
        # Setup signal handler for graceful interruption
        signal.signal(signal.SIGINT, self._signal_handler)

    def _json_default(o):
        """Best-effort converter for non-JSON-serializable SDK objects."""
        # MapComposite and other mapping-like objects
        try:
            return dict(o)
        except Exception:
            pass
        # Protobuf message/struct → dict
        try:
            from google.protobuf.json_format import MessageToDict
            return MessageToDict(o, preserving_proto_field_name=True)
        except Exception:
            pass
        # Last resort: string
        return str(o)

    def _signal_handler(self, signum, frame):
        """Handle interrupt signal gracefully."""
        self._interrupted = True
        print("\n\n[WARNING] Processing interrupted by user. Saving checkpoint...")
        logging.info("Processing interrupted by user")
    
    def _log_token_usage(self, response, chunk_num=None):
        """Log and accumulate token usage from API response."""
        if not response or not response.usage:
            return
        
        usage = response.usage
        chunk_label = f"Chunk {chunk_num}" if chunk_num else "Request"
        
        # Accumulate tokens
        for key in ["prompt_tokens", "completion_tokens", "total_tokens", 
                   "thoughts_tokens", "reasoning_tokens", "cached_tokens"]:
            if key in usage:
                self.stats["total_tokens"][key] += usage[key]
        
        # Log current chunk usage
        usage_msg = f"\n[TOKEN USAGE] {chunk_label}:"
        usage_msg += f"\n  - Prompt: {usage.get('prompt_tokens', 0):,} tokens"
        usage_msg += f"\n  - Completion: {usage.get('completion_tokens', 0):,} tokens"
        
        # Add special tokens if present
        if usage.get('thoughts_tokens'):
            usage_msg += f"\n  - Thoughts: {usage['thoughts_tokens']:,} tokens"
        if usage.get('reasoning_tokens'):
            usage_msg += f"\n  - Reasoning: {usage['reasoning_tokens']:,} tokens"
        if usage.get('cached_tokens'):
            usage_msg += f"\n  - Cached: {usage['cached_tokens']:,} tokens"
        
        usage_msg += f"\n  - Total: {usage.get('total_tokens', 0):,} tokens"
        
        # Only print if not using tqdm (to avoid cluttering progress bar)
        if not TQDM_AVAILABLE:
            print(usage_msg)
        else:
            # Log to file instead when using progress bar
            self.logger.info(usage_msg.replace('\n', ' '))
    
    def _display_final_token_summary(self):
        """Display final token usage summary."""
        tokens = self.stats["total_tokens"]
        
        # Skip if no tokens were used
        if tokens["total_tokens"] == 0:
            return
        
        print(f"\n[FINAL TOKEN USAGE]")
        print(f"  Total API Calls: {self.stats['api_calls']}")
        print(f"  Total Prompt Tokens: {tokens['prompt_tokens']:,}")
        print(f"  Total Completion Tokens: {tokens['completion_tokens']:,}")
        
        # Show special tokens if present
        if tokens.get('thoughts_tokens', 0) > 0:
            print(f"  Total Thoughts Tokens: {tokens['thoughts_tokens']:,}")
        if tokens.get('reasoning_tokens', 0) > 0:
            print(f"  Total Reasoning Tokens: {tokens['reasoning_tokens']:,}")
        if tokens.get('cached_tokens', 0) > 0:
            print(f"  Total Cached Tokens: {tokens['cached_tokens']:,}")
        
        print(f"  Grand Total: {tokens['total_tokens']:,} tokens")
        
        # Estimate costs if we have the model name
        if hasattr(self, 'llm_provider') and hasattr(self.llm_provider, 'model_name'):
            model = self.llm_provider.model_name.lower()
            
            # Basic cost estimates (these are approximate and should be updated)
            cost_per_1k = {
                'gpt-4o': {'input': 0.005, 'output': 0.015},
                'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
                'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
                'gemini-1.5-flash': {'input': 0.000075, 'output': 0.0003},
                'gemini-1.5-pro': {'input': 0.00125, 'output': 0.005},
                'gemini-2.5-flash': {'input': 0.000010, 'output': 0.000040},  # $0.10/$0.40 per 1M
            }
            
            # Find matching cost model
            for model_key, costs in cost_per_1k.items():
                if model_key in model:
                    input_cost = (tokens['prompt_tokens'] / 1000) * costs['input']
                    output_cost = (tokens['completion_tokens'] / 1000) * costs['output']
                    total_cost = input_cost + output_cost
                    print(f"\n  Estimated Cost: ${total_cost:.4f}")
                    print(f"    (Input: ${input_cost:.4f}, Output: ${output_cost:.4f})")
                    break
    
    def _setup_provider(self, provider: str, **kwargs) -> LLMProvider:
        """
        Set up the LLM provider.
        
        Args:
            provider: Provider name
            **kwargs: Provider configuration
            
        Returns:
            Configured LLMProvider instance
        """
        try:
            llm = get_provider(provider, **kwargs)
            print(f"[OK] Initialized {provider} provider: {llm.get_provider_info()}")
            return llm
        except Exception as e:
            print(f"[ERROR] Failed to initialize provider {provider}: {e}")
            raise

    def _save_checkpoint(
        self,
        chunk_idx: int,
        total_chunks: int | None = None,
        records_so_far=None,
        chunk_records=None,
        llm_response=None,
    ):
        """
        Persist a JSON-serializable checkpoint.
        Compatible with calls like:
        _save_checkpoint(chunk_idx, total_chunks, all_records, response.records)
        """

        import time, json

        # -------- safe serializer (backstop) ----------
        def _json_default(o):
            # MapComposite / mapping-like
            try:
                return {k: _json_default(v) for k, v in dict(o).items()}
            except Exception:
                pass
            # List/tuple
            if isinstance(o, (list, tuple)):
                return [_json_default(x) for x in o]
            # Protobuf
            try:
                from google.protobuf.json_format import MessageToDict
                return MessageToDict(o, preserving_proto_field_name=True)
            except Exception:
                pass
            # Primitives or last resort string
            try:
                json.dumps(o)
                return o
            except Exception:
                return str(o)

        # -------- normalize records to plain dicts -----
        def _normalize_list(lst):
            if lst is None:
                return []
            norm = []
            for item in lst:
                # mapping-like
                try:
                    as_dict = dict(item)
                    item = as_dict
                except Exception:
                    pass
                # protobuf → dict
                if not isinstance(item, (dict, list, tuple, str, int, float, bool, type(None))):
                    try:
                        from google.protobuf.json_format import MessageToDict
                        item = MessageToDict(item, preserving_proto_field_name=True)
                    except Exception:
                        # leave as is; default serializer will stringify
                        pass
                norm.append(item)
            return norm

        # provider info
        provider_info = {}
        try:
            if getattr(self, "provider", None) and hasattr(self.provider, "get_provider_info"):
                provider_info = self.provider.get_provider_info()
        except Exception as e:
            if getattr(self, "logger", None):
                self.logger.debug(f"get_provider_info failed: {e}")

        # records_so_far
        try:
            if records_so_far is None:
                records_so_far = list(getattr(self, "records", []))
            else:
                records_so_far = list(records_so_far)
        except Exception:
            records_so_far = []
        records_so_far = _normalize_list(records_so_far)

        # chunk_records
        if chunk_records is not None:
            chunk_records = _normalize_list(list(chunk_records))

        # llm_response (slim dict; raw_response kept as string only)
        lr = llm_response
        if hasattr(lr, "__dict__"):
            lr_dict = {
                "success": getattr(lr, "success", None),
                "error": getattr(lr, "error", None),
                "usage": getattr(lr, "usage", None),
                "records": _normalize_list(getattr(lr, "records", None)),
                "raw_response": getattr(lr, "raw_response", None)
                if isinstance(getattr(lr, "raw_response", None), str)
                else (str(getattr(lr, "raw_response", "")) if getattr(lr, "raw_response", None) is not None else None),
            }
        elif isinstance(lr, dict):
            lr_dict = {
                "success": lr.get("success"),
                "error": lr.get("error"),
                "usage": lr.get("usage"),
                "records": _normalize_list(lr.get("records")),
                "raw_response": lr.get("raw_response") if isinstance(lr.get("raw_response"), str)
                else (str(lr.get("raw_response")) if lr.get("raw_response") is not None else None),
            }
        else:
            lr_dict = None

        checkpoint_data = {
            "stage": "genealogy_parsing",
            "chunk_index": int(chunk_idx),
            "total_chunks": int(total_chunks) if total_chunks is not None else None,
            "provider": provider_info,
            "last_response": lr_dict,
            "totals": {"records_so_far": len(records_so_far)},
            "timestamp": time.time(),
            "token_usage": getattr(self, "stats", {}).get("total_tokens", {}),
            "api_calls": getattr(self, "stats", {}).get("api_calls", 0),
        }

        enc = getattr(self, "encoding", "utf-8")
        ckpt_path = getattr(self, "checkpoint_file", None)
        recs_ckpt_path = getattr(self, "records_checkpoint_file", None)

        # main checkpoint
        try:
            if ckpt_path:
                with open(ckpt_path, "w", encoding=enc) as f:
                    json.dump(checkpoint_data, f, ensure_ascii=False, indent=2, default=_json_default)
        except Exception as e:
            if getattr(self, "logger", None):
                self.logger.error(f"Failed to save checkpoint: {e}")

        # records-so-far
        try:
            if recs_ckpt_path:
                with open(recs_ckpt_path, "w", encoding=enc) as f:
                    json.dump(records_so_far, f, ensure_ascii=False, indent=2, default=_json_default)
        except Exception as e:
            if getattr(self, "logger", None):
                self.logger.error(f"Failed to save records checkpoint: {e}")

        # per-chunk file (only save when we have valid chunk_records)
        try:
            if getattr(self, "save_intermediate", False) and chunk_records is not None:
                out_path = getattr(self, "output_file", None)
                if out_path is not None:
                    # Create chunks directory if it doesn't exist
                    chunks_dir = out_path.parent / "chunks"
                    chunks_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save chunk file to chunks directory
                    inter_file = chunks_dir / f"{out_path.stem}_chunk_{chunk_idx:03d}.json"
                    # Only save the chunk_records, never the cumulative records_so_far
                    with open(inter_file, "w", encoding=enc) as f:
                        json.dump(chunk_records, f, ensure_ascii=False, indent=2, default=_json_default)
        except Exception as e:
            if getattr(self, "logger", None):
                self.logger.error(f"Failed to save intermediate chunk file: {e}")

    def _load_checkpoint(self) -> Optional[Dict]:
        """Load checkpoint if it exists and resume is enabled."""
        if not self.resume:
            return None
            
        if not self.checkpoint_file.exists() or not self.records_checkpoint_file.exists():
            logging.info("No checkpoint found to resume from")
            return None
        
        try:
            with open(self.checkpoint_file, 'r', encoding=self.encoding) as f:
                checkpoint = json.load(f)
            
            with open(self.records_checkpoint_file, 'r', encoding=self.encoding) as f:
                records = json.load(f)
            
            checkpoint["records"] = records
            
            logging.info(f"Checkpoint loaded: resuming from chunk {checkpoint['chunk_index']}/{checkpoint['total_chunks']}")
            logging.info(f"  Previous records: {len(records)}")
            logging.info(f"  Previous API calls: {checkpoint.get('api_calls', 0)}")
            
            return checkpoint
            
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            return None

    def _clear_checkpoint(self):
        """Clear checkpoint files after successful completion."""
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
            if self.records_checkpoint_file.exists():
                self.records_checkpoint_file.unlink()
            # Checkpoint files cleared silently
        except Exception as e:
            # Failed to clear checkpoint files silently
            pass

    def _post_process_records(self, raw_records: List[Dict]) -> List[Dict]:
        """
        Merges supplementary records into their parent records.
        This moves the complex "update" logic from the AI to Python.
        Also filters out records marked with skip=true.
        """
        if not raw_records:
            return []

        # Filter out skipped records first
        non_skipped = [r for r in raw_records if not r.get('skip', False)]
        logging.info(f"Filtered {len(raw_records) - len(non_skipped)} skipped records out of {len(raw_records)} total")

        final_records: List[Dict] = []
        for i, record_data in enumerate(non_skipped):
            try:
                # Handle is_update_for_previous flag
                is_update = record_data.pop('is_update_for_previous', False)
                # Remove skip field as it's no longer needed
                record_data.pop('skip', None)
                
                # Process children field
                children_data = record_data.get('children', [])
                children_list = []
                for child in children_data:
                    if isinstance(child, dict):
                        # Convert order to int if it's a string number
                        order_val = child.get('order')
                        if isinstance(order_val, str) and order_val.isdigit():
                            order_val = int(order_val)
                        elif not isinstance(order_val, int):
                            order_val = None
                        
                        children_list.append({
                            'order': order_val,
                            'name': child.get('name'),
                            'sex': child.get('sex')
                        })
                
                # Convert birth_order to int if it's a string number
                birth_order_val = record_data.get('birth_order')
                if isinstance(birth_order_val, str) and birth_order_val.isdigit():
                    birth_order_val = int(birth_order_val)
                elif not isinstance(birth_order_val, int):
                    birth_order_val = None
                
                # Create processed record with proper None values for nullable fields
                processed_data = {
                    'name': record_data.get('name'),
                    'sex': record_data.get('sex'),
                    'father': record_data.get('father'),
                    'birth_order': birth_order_val,
                    'courtesy': record_data.get('courtesy'),
                    'birth_time': record_data.get('birth_time'),
                    'death_time': record_data.get('death_time'),
                    'children': children_list,
                    'info': record_data.get('info'),
                    'original_text': record_data.get('original_text', ''),
                    'note': record_data.get('note')
                }

                # Check if merge_updates is enabled
                
                if is_update and final_records and self.merge_updates:
                    # This record contains info for the previous person. Merge it.
                    last_record = final_records[-1]
                    
                    # Merge info field
                    if processed_data['info']:
                        if last_record['info']:
                            last_record['info'] += " " + processed_data['info']
                        else:
                            last_record['info'] = processed_data['info']
                    
                    # Merge original_text
                    last_record['original_text'] += "\n" + processed_data['original_text']
                    
                    # Merge note field
                    if processed_data['note']:
                        if last_record['note']:
                            last_record['note'] += " | " + processed_data['note']
                        else:
                            last_record['note'] = processed_data['note']
                    
                    # Update sex if it was missing
                    if not last_record.get('sex') and processed_data.get('sex'):
                        last_record['sex'] = processed_data['sex']
                    
                    # Update birth_time if it was missing
                    if not last_record.get('birth_time') and processed_data.get('birth_time'):
                        last_record['birth_time'] = processed_data['birth_time']
                    
                    # Update death_time if it was missing
                    if not last_record.get('death_time') and processed_data.get('death_time'):
                        last_record['death_time'] = processed_data['death_time']
                else:
                    # This is a new person, so we append the processed dictionary.
                    # If merge_updates is disabled, even update records become separate entries
                    final_records.append(processed_data)
                    
            except Exception as e:
                # Log errors to stats, not to console
                self.stats["errors"].append(f"Record {i}: {e}")
        
        # Post-processing complete
        return final_records

    def parse_genealogy(self, chunk_size=20, context_size=10):
        """Main method to parse the genealogy text with context preservation and checkpoint support."""
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file '{self.input_file}' not found")

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n[INFO] Starting genealogy parsing from '{self.input_file}'")

        with open(self.input_file, 'r', encoding=self.encoding) as f:
            lines = f.readlines()
        
        self.stats["lines_processed"] = len(lines)
        print(f"[INFO] Read {self.stats['lines_processed']} lines from file")
        print(f"[INFO] Context size: {context_size} lines\n")

        # Load checkpoint if resuming
        checkpoint = self._load_checkpoint()
        if checkpoint:
            all_records = checkpoint["records"]
            start_chunk = checkpoint["chunk_index"] + 1
            self.stats["api_calls"] = checkpoint.get("api_calls", 0)
            
            # Restore token usage stats if available
            if "token_usage" in checkpoint:
                self.stats["total_tokens"].update(checkpoint["token_usage"])
            
            print(f"[OK] Resuming from checkpoint: chunk {start_chunk}")
            print(f"   Previous records: {len(all_records)}")
            print(f"   Previous API calls: {checkpoint.get('api_calls', 0)}")
            
            # Show token usage if available
            if checkpoint.get("token_usage") and checkpoint["token_usage"].get("total_tokens"):
                print(f"   Previous tokens used: {checkpoint['token_usage']['total_tokens']:,}")
            print()
        else:
            all_records = []
            start_chunk = 0

        # Process in chunks with context preservation
        if len(lines) > chunk_size:
            total_chunks = (len(lines) - 1) // chunk_size + 1
            print(f"[INFO] Processing {total_chunks} chunks ({chunk_size} lines each)\n")
            
            # Create progress bar if tqdm is available
            if TQDM_AVAILABLE:
                chunk_range = tqdm(range(start_chunk, total_chunks), 
                                 initial=start_chunk, total=total_chunks,
                                 desc="Processing chunks", unit="chunk",
                                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
            else:
                chunk_range = range(start_chunk, total_chunks)
            
            for chunk_idx in chunk_range:
                if self._interrupted:
                    print("\n[WARNING] Processing interrupted. Saving progress...")
                    self._save_checkpoint(chunk_idx - 1, total_chunks, all_records, None)
                    return {"lines_processed": self.stats["lines_processed"],
                            "records_created": len(all_records),
                            "api_calls": self.stats["api_calls"],
                            "checkpoint_saved": True,
                            "interrupted": True}
                i = chunk_idx * chunk_size
                chunk_num = chunk_idx + 1
                
                if i == 0:
                    # First chunk - no context needed
                    chunk_lines = lines[0:min(chunk_size, len(lines))]
                    chunk_text = ''.join(chunk_lines)
                    if not TQDM_AVAILABLE:
                        print(f"\n[PROCESSING] Chunk {chunk_num}/{total_chunks}: lines 1-{len(chunk_lines)}")
                    
                    response = self.llm_provider.parse_genealogy(
                        text=chunk_text,
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt_template=USER_PROMPT_TEMPLATE,
                        use_function_calling=self.use_function_calling
                    )
                    self._log_token_usage(response, chunk_num)
                else:
                    # Include context from previous chunk
                    context_start = max(0, i - context_size)
                    context_lines = lines[context_start:i]
                    new_lines = lines[i:min(i + chunk_size, len(lines))]
                    
                    context_text = ''.join(context_lines)
                    new_text = ''.join(new_lines)
                    
                    if not TQDM_AVAILABLE:
                        print(f"\n[PROCESSING] Chunk {chunk_num}/{total_chunks}: lines {i+1}-{min(i+chunk_size, len(lines))} (with {len(context_lines)} context lines)")
                    
                    response = self.llm_provider.parse_with_context(
                        context_text=context_text,
                        new_text=new_text,
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt_template=USER_PROMPT_WITH_CONTEXT,
                        use_function_calling=self.use_function_calling
                    )
                    self._log_token_usage(response, chunk_num)
                
                self.stats["api_calls"] += 1
                # Salvage: if the call FAILED OR returned NO records, retry without context using plain JSON mode
                if (not response.success) or (not response.records):
                    try:
                        salvage_text = chunk_text if i == 0 else new_text
                    except NameError:
                        salvage_text = new_text if chunk_idx > 0 else ''.join(lines[0:min(chunk_size, len(lines))])
                    self.logger.warning(f"Chunk {chunk_num}: empty/failed; attempting salvage without context...")
                    salvage_resp = self.llm_provider.parse_genealogy(
                        text=salvage_text,
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt_template=USER_PROMPT_TEMPLATE,
                        use_function_calling=False
                    )
                    self._log_token_usage(salvage_resp, f"{chunk_num} (salvage)")
                    self.stats["api_calls"] += 1
                    if salvage_resp.success and salvage_resp.records:
                        response = salvage_resp
                        self.logger.info(f"Chunk {chunk_num}: salvage succeeded with {len(response.records)} records")
                    else:
                        self.logger.warning(f"Chunk {chunk_num}: salvage attempt returned empty or failed")
                
                # Check if API call failed
                if not response.success:
                    error_msg = str(response.error)
                    
                    # Check for MAX_TOKENS error
                    if "finish_reason" in error_msg and "is 2" in error_msg:
                        print(f"\n[ERROR] Chunk {chunk_num} too large for model token limit")
                        print(f"[SOLUTION] Reduce chunk_size in config.yaml (current: {chunk_size} lines)")
                        print("Recommended: chunk_size: 2-3 lines for complex text")
                    else:
                        print(f"\n[ERROR] Chunk {chunk_num} failed: {response.error}")
                    
                    self._save_checkpoint(chunk_idx - 1 if chunk_idx > 0 else 0, total_chunks, all_records, None)
                    print("\n[INFO] Progress saved. You can resume processing later with --resume flag")
                    return {"lines_processed": self.stats["lines_processed"],
                            "records_created": len(all_records),
                            "api_calls": self.stats["api_calls"],
                            "checkpoint_saved": True}
                
                # Add deduplication for GPT-5 which may include context records
                if chunk_idx > 0 and response.records:
                    # For subsequent chunks, check for duplicates
                    existing_texts = {r.get('original_text', '') for r in all_records if r.get('original_text')}
                    new_records = []
                    duplicates = 0
                    
                    for record in response.records:
                        if record.get('original_text') and record['original_text'] in existing_texts:
                            duplicates += 1
                            continue  # Skip duplicate
                        new_records.append(record)
                    
                    if duplicates > 0:
                        self.logger.warning(f"Chunk {chunk_num}: Filtered {duplicates} duplicate records")
                    
                    all_records.extend(new_records)
                    records_added = len(new_records)
                else:
                    # First chunk - no duplicates possible
                    all_records.extend(response.records)
                    records_added = len(response.records) if response.records else 0
                
                # Update progress display
                if not TQDM_AVAILABLE:
                    print(f"   [OK] Found {records_added} new records (Total: {len(all_records)})")
                
                # Save checkpoint after each successful chunk
                # Pass current chunk records separately for intermediate file
                self._save_checkpoint(chunk_idx, total_chunks, all_records, response.records)
                
                # Small delay between chunks to avoid rate limits
                if chunk_num < total_chunks:
                    time.sleep(1)
        else:
            # Process the entire text in one go
            print(f"\n[INFO] Processing entire file in single request...")
            full_text = ''.join(lines)
            response = self.llm_provider.parse_genealogy(
                text=full_text,
                system_prompt=SYSTEM_PROMPT,
                user_prompt_template=USER_PROMPT_TEMPLATE,
                use_function_calling=self.use_function_calling
            )
            self._log_token_usage(response)
            self.stats["api_calls"] += 1
            
            if response.success:
                all_records = response.records
                print(f"   [OK] Found {len(all_records)} records")
            else:
                print(f"\n[ERROR] Parsing failed: {response.error}")
                all_records = []
        
        # Check if interrupted
        if self._interrupted:
            print("\n[WARNING] Processing was interrupted.")
            return {"lines_processed": self.stats["lines_processed"],
                    "records_created": len(all_records),
                    "api_calls": self.stats["api_calls"],
                    "checkpoint_saved": True,
                    "interrupted": True}
        
        raw_records = all_records
        
        # Perform post-processing to merge update lines
        print("\n[PROCESSING] Post-processing records...")
        final_records = self._post_process_records(raw_records)
        self.stats["records_created"] = len(final_records)
        print(f"   [OK] Merged updates, final count: {len(final_records)} records")

        # Save final output
        print(f"\n[SAVING] Saving final output...")
        with open(self.output_file, 'w', encoding=self.encoding) as f:
            json.dump(final_records, f, ensure_ascii=False, indent=2)
        print(f"   [OK] Output saved to '{self.output_file}'")
        
        # Remove intermediate chunk files if save_intermediate is enabled
        if self.save_intermediate and len(lines) > chunk_size:
            print("\n[CLEANUP] Cleaning up intermediate files...")
            for chunk_file in self.output_file.parent.glob(f"{self.output_file.stem}_chunk_*.json"):
                try:
                    chunk_file.unlink()
                except:
                    pass
        
        if self.stats["errors"]:
            error_file = self.output_file.with_suffix('.errors.txt')
            with open(error_file, 'w', encoding=self.encoding) as f:
                for error in self.stats["errors"]:
                    f.write(f"- {error}\n")
            print(f"   [WARNING] Errors saved to '{error_file}'")
        
        # Clear checkpoint on successful completion
        self._clear_checkpoint()
        
        # Display final token usage summary
        self._display_final_token_summary()
        
        print(f"\n[COMPLETE] Parsing complete!")
        print(f"   Total records: {self.stats['records_created']}")
        print(f"   API calls: {self.stats['api_calls']}")
        return self.stats


def main():
    """Command-line interface for the genealogy parser."""
    parser = argparse.ArgumentParser(
        description="Parse genealogy text using AI to extract structured family data.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input_file", help="Input file to parse (e.g., cleaned_text.txt)")
    parser.add_argument("-o", "--output", default="genealogy_data.json", help="Output JSON file")
    parser.add_argument("-p", "--provider", default="litellm", 
                       choices=['openai', 'gemini', 'litellm'],
                       help="LLM provider to use")
    parser.add_argument("-m", "--model", default="gpt-4o-mini", help="Model name")
    parser.add_argument("-k", "--api-key", help="API key (or use environment variable)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose DEBUG logging")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--chunk-size", type=int, default=20, help="Lines per chunk (default: 20)")
    parser.add_argument("--context-size", type=int, default=10, help="Context lines (default: 10)")
    parser.add_argument("--checkpoint-dir", help="Directory for checkpoint files")
    parser.add_argument("--no-intermediate", action="store_true", help="Don't save intermediate results")
    parser.add_argument("--no-function-calling", action="store_true", help="Disable function calling")
    parser.add_argument("--temperature", type=float, default=0.2, help="Model temperature (0.0-2.0)")
    
    args = parser.parse_args()
    
    # Set up minimal logging only for errors
    logging.basicConfig(
        level=logging.ERROR,
        format='%(levelname)s: %(message)s'
    )
    
    try:
        parser_obj = GenealogyParser(
            input_file=args.input_file,
            output_file=args.output,
            provider=args.provider,
            model_name=args.model,
            api_key=args.api_key,
            temperature=args.temperature,
            use_function_calling=not args.no_function_calling,
            resume=args.resume,
            checkpoint_dir=args.checkpoint_dir,
            save_intermediate=not args.no_intermediate
        )
        stats = parser_obj.parse_genealogy(
            chunk_size=args.chunk_size,
            context_size=args.context_size
        )
        
        if not stats.get('interrupted'):
            print("\n" + "=" * 50)
            print("📊 Parsing Summary")
            print("=" * 50)
            print(f"  Lines processed: {stats['lines_processed']:,}")
            print(f"  Final records created: {stats['records_created']:,}")
            print(f"  API calls made: {stats['api_calls']}")
            print(f"  Errors encountered: {len(stats.get('errors', []))}")
            
            if stats.get('checkpoint_saved') and not stats.get('interrupted'):
                print("\n[WARNING] Processing paused due to API limits.")
                print("  Run with --resume to continue from checkpoint")
            else:
                print(f"\n📄 Output saved to: {parser_obj.output_file}")
        
    except KeyboardInterrupt:
        print("\n\n[WARNING] Processing interrupted by user")
        print("Progress has been saved. Use --resume to continue.")
        return 0
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())