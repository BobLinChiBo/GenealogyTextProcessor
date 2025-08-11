#!/usr/bin/env python3
"""
Parallel genealogy parser using asyncio for concurrent chunk processing.

This module processes multiple text chunks simultaneously for faster parsing,
using text-based context overlap instead of sequential dependencies.
"""

import asyncio
import json
import logging
import signal
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import hashlib

# Progress bar support (optional)
try:
    from tqdm.asyncio import tqdm as async_tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Import LLM providers
import sys
sys.path.append(str(Path(__file__).parent.parent))
from llm import get_provider, LLMProvider, LLMResponse

# Import centralized prompts
from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, USER_PROMPT_WITH_CONTEXT


@dataclass
class ChunkTask:
    """Represents a chunk processing task."""
    chunk_id: int
    start_line: int
    end_line: int
    text_lines: List[str]
    context_lines: List[str] = field(default_factory=list)
    
    @property
    def full_text(self) -> str:
        """Get the full text for this chunk (context + new lines)."""
        return ''.join(self.context_lines + self.text_lines)
    
    @property
    def new_text(self) -> str:
        """Get just the new text lines for this chunk."""
        return ''.join(self.text_lines)
    
    @property
    def context_text(self) -> str:
        """Get just the context text lines."""
        return ''.join(self.context_lines)


class ParallelGenealogyParser:
    """Handles parallel AI-powered parsing of genealogy text."""

    # Prompts are now imported from centralized module at the top of the file

    def __init__(self,
                 input_file: str,
                 output_file: str,
                 provider: str = "litellm",
                 model_name: str = None,
                 api_key: str = None,
                 temperature: float = 0.2,
                 max_workers: int = 5,
                 context_size: int = 5,  # Standardized with parser_v3
                 use_function_calling: bool = True,
                 max_retries: int = 3,
                 retry_delay: int = 5,
                 encoding: str = 'utf-8',
                 save_intermediate: bool = True,
                 requests_per_minute: int = 50,
                 auto_adjust_workers: bool = True,
                 slow_start: bool = True):
        """
        Initialize the parallel parser.
        
        Args:
            input_file: Path to cleaned text file
            output_file: Path for output JSON
            provider: LLM provider name
            model_name: Model to use
            api_key: API key for the provider
            temperature: Model temperature
            max_workers: Maximum concurrent workers
            context_size: Number of context lines from previous chunk (same as parser_v3)
            use_function_calling: Whether to use function calling
            max_retries: Max retry attempts for failed chunks
            retry_delay: Delay between retries
            encoding: File encoding
            save_intermediate: Save intermediate results
            requests_per_minute: Maximum requests per minute (rate limiting)
            auto_adjust_workers: Automatically reduce workers on rate limits
            slow_start: Start with 1 worker and gradually increase
        """
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.encoding = encoding
        self.use_function_calling = use_function_calling
        self.temperature = temperature
        self.save_intermediate = save_intermediate
        self.initial_max_workers = max_workers  # Store original value
        self.context_size = context_size  # Renamed from overlap_lines for consistency
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.requests_per_minute = requests_per_minute
        self.auto_adjust_workers = auto_adjust_workers
        self.slow_start = slow_start
        
        # Slow start: begin with 1 worker
        if slow_start:
            self.max_workers = 1
            logging.info(f"Slow start enabled: beginning with 1 worker (target: {self.initial_max_workers})")
        else:
            self.max_workers = max_workers
        
        # Setup LLM provider
        provider_config = {
            'model_name': model_name,
            'api_key': api_key,
            'temperature': temperature
        }
        
        # GPT-5 specific adjustments
        if model_name and 'gpt-5' in model_name.lower():
            # GPT-5 is slower, so reduce rate limits and increase timeouts
            self.requests_per_minute = min(requests_per_minute, 20)  # Reduce from default 50
            self.tokens_per_minute = 20000  # Reduce from 25000
            self.min_request_interval = 3.0  # Increase from 1.0 to 3.0 seconds
            # GPT-5 doesn't support function calling properly
            self.use_function_calling = False
            logging.info(f"GPT-5 detected: adjusted rate limits (RPM: {self.requests_per_minute}, TPM: {self.tokens_per_minute}, min interval: {self.min_request_interval}s)")
            logging.info(f"GPT-5 detected: disabled function calling")
        
        self.llm_provider = self._setup_provider(
            provider=provider,
            max_retries=max_retries,
            retry_delay=retry_delay,
            **provider_config
        )
        
        # Stats tracking
        self.stats = {
            "lines_processed": 0,
            "records_created": 0,
            "api_calls": 0,
            "errors": [],
            "chunks_processed": 0,
            "chunks_failed": 0
        }
        
        # Semaphore for rate limiting
        self.semaphore = asyncio.Semaphore(max_workers)
        
        # Rate limiting tracking
        self.request_times = []  # Track request timestamps for rate limiting
        self.rate_limit_hits = 0  # Count rate limit hits for auto-adjustment
        self.last_request_time = 0  # Track last request time for minimum interval
        self.min_request_interval = 1.0  # Minimum seconds between ANY API calls (can be configured)
        self.token_usage = []  # Track token usage with timestamps
        self.tokens_per_minute = 25000  # Conservative TPM limit (can be configured)
        self.successful_chunks = 0  # Track successful chunks for slow start
        
        # Interrupt handling
        self._interrupted = False
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle interrupt signal gracefully."""
        self._interrupted = True
        print("\n\n[WARNING] Processing interrupted by user...")
        logging.info("Processing interrupted by user")

    def _setup_provider(self, provider: str, **kwargs) -> LLMProvider:
        """Set up the LLM provider."""
        try:
            llm = get_provider(provider, **kwargs)
            print(f"[OK] Initialized {provider} provider: {llm.get_provider_info()}")
            return llm
        except Exception as e:
            print(f"[ERROR] Failed to initialize provider {provider}: {e}")
            raise

    def _prepare_chunks(self, lines: List[str], chunk_size: int) -> List[ChunkTask]:
        """
        Prepare chunks with text-based overlap for parallel processing.
        
        Args:
            lines: All text lines
            chunk_size: Lines per chunk
            
        Returns:
            List of ChunkTask objects ready for processing
        """
        chunks = []
        total_lines = len(lines)
        
        for i in range(0, total_lines, chunk_size):
            # Determine chunk boundaries
            start_line = i
            end_line = min(i + chunk_size, total_lines)
            
            # Get context lines from previous chunk (same logic as parser_v3)
            context_start = max(0, start_line - self.context_size)
            context_lines = lines[context_start:start_line] if start_line > 0 else []
            
            # Get main chunk lines
            text_lines = lines[start_line:end_line]
            
            # Create chunk task
            chunk = ChunkTask(
                chunk_id=len(chunks),
                start_line=start_line,
                end_line=end_line,
                text_lines=text_lines,
                context_lines=context_lines
            )
            chunks.append(chunk)
        
        return chunks

    async def _enforce_rate_limit(self, estimated_tokens: int = 1500):
        """Enforce rate limiting based on requests per minute and tokens per minute.
        
        Args:
            estimated_tokens: Estimated tokens for the next request
        """
        current_time = time.time()
        
        # Enforce minimum interval between ANY requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last
            logging.debug(f"Enforcing minimum interval. Waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
            current_time = time.time()
        
        # Remove timestamps older than 1 minute
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        self.token_usage = [(t, tokens) for t, tokens in self.token_usage if current_time - t < 60]
        
        # Check request rate limit
        if self.requests_per_minute > 0 and len(self.request_times) >= self.requests_per_minute:
            # Calculate how long to wait
            oldest_request = self.request_times[0]
            wait_time = 60 - (current_time - oldest_request) + 0.5  # Add buffer
            
            if wait_time > 0:
                logging.info(f"Request rate limit reached ({self.requests_per_minute} req/min). Waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                current_time = time.time()
        
        # Check token rate limit
        current_tokens = sum(tokens for _, tokens in self.token_usage)
        if current_tokens + estimated_tokens > self.tokens_per_minute:
            # Need to wait for some tokens to fall out of the window
            tokens_to_free = (current_tokens + estimated_tokens) - self.tokens_per_minute
            
            # Find when enough tokens will be freed
            freed_tokens = 0
            wait_until = current_time
            for timestamp, tokens in self.token_usage:
                freed_tokens += tokens
                if freed_tokens >= tokens_to_free:
                    wait_until = timestamp + 60
                    break
            
            wait_time = max(0, wait_until - current_time) + 1  # Add 1s buffer
            if wait_time > 0:
                logging.info(f"Token rate limit approaching ({current_tokens + estimated_tokens}/{self.tokens_per_minute} TPM). Waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                current_time = time.time()
        
        # Record this request
        self.request_times.append(current_time)
        self.token_usage.append((current_time, estimated_tokens))
        self.last_request_time = current_time
    
    async def _adjust_workers_on_rate_limit(self):
        """Dynamically reduce workers when hitting rate limits."""
        if not self.auto_adjust_workers:
            return
        
        self.rate_limit_hits += 1
        
        # Immediately reduce to 1 worker on first rate limit
        if self.max_workers > 1:
            new_max_workers = 1
            logging.info(f"Rate limit hit! Reducing max_workers from {self.max_workers} to {new_max_workers}")
            
            # Update the semaphore
            self.max_workers = new_max_workers
            self.semaphore = asyncio.Semaphore(new_max_workers)
            
            # Reset counters
            self.rate_limit_hits = 0
            self.successful_chunks = 0  # Reset success counter
    
    async def _increase_workers_on_success(self):
        """Gradually increase workers after successful chunks (slow start)."""
        if not self.slow_start or not self.auto_adjust_workers:
            return
        
        self.successful_chunks += 1
        
        # Increase workers after every 5 successful chunks, up to the initial max
        if self.successful_chunks >= 5 and self.max_workers < self.initial_max_workers:
            new_max_workers = min(self.max_workers + 1, self.initial_max_workers)
            logging.info(f"Increasing max_workers from {self.max_workers} to {new_max_workers} after {self.successful_chunks} successful chunks")
            
            # Update the semaphore
            self.max_workers = new_max_workers
            self.semaphore = asyncio.Semaphore(new_max_workers)
            
            # Reset the counter
            self.successful_chunks = 0
    
    async def _process_chunk(self, chunk: ChunkTask, pbar=None, retry_count: int = 0) -> Tuple[int, List[Dict]]:
        """
        Process a single chunk asynchronously with retry logic and rate limiting.
        
        Args:
            chunk: ChunkTask to process
            pbar: Optional progress bar
            retry_count: Current retry attempt number
            
        Returns:
            Tuple of (chunk_id, records)
        """
        async with self.semaphore:  # Concurrency limiting
            if self._interrupted:
                return (chunk.chunk_id, [])
            
            # Estimate tokens for this chunk (rough estimate: 4 chars = 1 token)
            estimated_tokens = len(chunk.full_text) // 4 + 500  # Add buffer for system prompt
            
            # Enforce rate limiting with token estimate
            await self._enforce_rate_limit(estimated_tokens)
            
            try:
                # Log chunk processing
                if retry_count == 0:
                    logging.info(f"Processing chunk {chunk.chunk_id}: lines {chunk.start_line+1}-{chunk.end_line}")
                else:
                    logging.info(f"Retrying chunk {chunk.chunk_id} (attempt {retry_count + 1}/{self.max_retries})")
                
                # Determine which prompt to use
                if chunk.chunk_id == 0:
                    # First chunk - no context
                    response = await self._async_parse(
                        text=chunk.new_text,
                        use_context=False
                    )
                else:
                    # Subsequent chunks - use context
                    response = await self._async_parse(
                        text=chunk.new_text,
                        context=chunk.context_text,
                        use_context=True
                    )
                
                self.stats["api_calls"] += 1
                
                # Update actual token usage if available
                if response.usage and response.usage.get('total_tokens'):
                    actual_tokens = response.usage['total_tokens']
                    # Update the last token usage entry with actual value
                    if self.token_usage:
                        self.token_usage[-1] = (self.token_usage[-1][0], actual_tokens)
                
                if response.success:
                    self.stats["chunks_processed"] += 1
                    if pbar:
                        pbar.update(1)
                    
                    # Gradually increase workers on success (slow start)
                    await self._increase_workers_on_success()
                    
                    # Save intermediate results if enabled
                    if self.save_intermediate:
                        await self._save_intermediate_chunk(chunk.chunk_id, response.records)
                    
                    return (chunk.chunk_id, response.records)
                else:
                    # Check if it's a rate limit error and we should retry
                    if ("rate_limit" in response.error.lower() or "429" in response.error) and retry_count < self.max_retries - 1:
                        # Adjust workers if auto-adjust is enabled
                        await self._adjust_workers_on_rate_limit()
                        
                        # Wait before retrying (the provider should have already waited, but add extra buffer)
                        await asyncio.sleep(2 ** retry_count)  # Exponential backoff: 1s, 2s, 4s
                        return await self._process_chunk(chunk, pbar, retry_count + 1)
                    
                    # For non-retryable errors or max retries reached
                    self.stats["chunks_failed"] += 1
                    self.stats["errors"].append(f"Chunk {chunk.chunk_id}: {response.error}")
                    logging.error(f"Chunk {chunk.chunk_id} failed: {response.error}")
                    
                    if pbar:
                        pbar.update(1)
                    
                    return (chunk.chunk_id, [])
                    
            except Exception as e:
                # For unexpected exceptions, try to retry if we haven't exceeded max retries
                if retry_count < self.max_retries - 1:
                    await asyncio.sleep(2 ** retry_count)
                    return await self._process_chunk(chunk, pbar, retry_count + 1)
                
                self.stats["chunks_failed"] += 1
                self.stats["errors"].append(f"Chunk {chunk.chunk_id}: {str(e)}")
                logging.error(f"Error processing chunk {chunk.chunk_id}: {e}")
                
                if pbar:
                    pbar.update(1)
                
                return (chunk.chunk_id, [])

    async def _async_parse(self, text: str, context: str = None, use_context: bool = False) -> LLMResponse:
        """
        Async wrapper for LLM parsing.
        
        This wraps the synchronous LLM provider call in an async executor.
        """
        loop = asyncio.get_event_loop()
        
        if use_context and context:
            # Parse with context (using named arguments like v3 for clarity)
            response = await loop.run_in_executor(
                None,
                lambda: self.llm_provider.parse_with_context(
                    context_text=context,
                    new_text=text,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt_template=USER_PROMPT_WITH_CONTEXT,
                    use_function_calling=self.use_function_calling
                )
            )
        else:
            # Parse without context (using named arguments like v3 for clarity)
            response = await loop.run_in_executor(
                None,
                lambda: self.llm_provider.parse_genealogy(
                    text=text,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt_template=USER_PROMPT_TEMPLATE,
                    use_function_calling=self.use_function_calling
                )
            )
        
        return response

    async def _save_intermediate_chunk(self, chunk_id: int, records: List[Dict]):
        """Save intermediate results for a chunk."""
        try:
            intermediate_file = self.output_file.parent / f"{self.output_file.stem}_chunk_{chunk_id:03d}.json"
            
            # Use async file writing
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: intermediate_file.write_text(
                    json.dumps(records, ensure_ascii=False, indent=2),
                    encoding=self.encoding
                )
            )
        except Exception as e:
            logging.error(f"Failed to save intermediate chunk {chunk_id}: {e}")

    def _deduplicate_records(self, all_records: List[Tuple[int, List[Dict]]]) -> List[Dict]:
        """
        Deduplicate and merge records from parallel chunks.
        Uses the same logic as parser_v3 for consistency.
        
        Args:
            all_records: List of (chunk_id, records) tuples
            
        Returns:
            Deduplicated list of records
        """
        # Sort by chunk ID to maintain order
        all_records.sort(key=lambda x: x[0])
        
        # Flatten records while maintaining order
        merged_records = []
        seen_original_texts = set()  # Track original_text for deduplication (same as parser_v3)
        
        for chunk_id, records in all_records:
            if chunk_id == 0:
                # First chunk - no duplicates possible
                merged_records.extend(records)
                for record in records:
                    if record.get('original_text'):
                        seen_original_texts.add(record['original_text'])
            else:
                # Subsequent chunks - check for duplicates (same logic as parser_v3)
                duplicates = 0
                for record in records:
                    original_text = record.get('original_text', '')
                    if original_text and original_text in seen_original_texts:
                        duplicates += 1
                        continue  # Skip duplicate
                    merged_records.append(record)
                    if original_text:
                        seen_original_texts.add(original_text)
                
                if duplicates > 0:
                    logging.warning(f"Chunk {chunk_id}: Filtered {duplicates} duplicate records")
        
        return merged_records

    def _post_process_records(self, raw_records: List[Dict]) -> List[Dict]:
        """
        Post-process records to merge updates into parent records.
        Also filters out records marked with skip=true.
        """
        if not raw_records:
            return []

        # Filter out skipped records first
        non_skipped = [r for r in raw_records if not r.get('skip', False)]
        logging.info(f"Filtered {len(raw_records) - len(non_skipped)} skipped records out of {len(raw_records)} total")

        final_records = []
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
                    'children': children_list,
                    'info': record_data.get('info'),
                    'original_text': record_data.get('original_text', ''),
                    'note': record_data.get('note')
                }

                # Check if merge_updates is enabled in config
                merge_updates = self.config.get('parser', {}).get('merge_updates', True)
                
                if is_update and final_records and merge_updates:
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
                else:
                    # Add as new record
                    # If merge_updates is disabled, even update records become separate entries
                    final_records.append(processed_data)
                    
            except Exception as e:
                self.stats["errors"].append(f"Record {i}: {e}")
        
        return final_records

    async def parse_genealogy_async(self, chunk_size: int = 20) -> Dict:
        """
        Main async method to parse genealogy text in parallel.
        
        Args:
            chunk_size: Lines per chunk
            
        Returns:
            Statistics dictionary
        """
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file '{self.input_file}' not found")

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[INFO] Starting parallel genealogy parsing")
        print(f"[INFO] Input: {self.input_file}")
        print(f"[INFO] Max workers: {self.max_workers}")
        print(f"[INFO] Chunk size: {chunk_size} lines")
        print(f"[INFO] Context size: {self.context_size} lines\n")

        # Read input file
        with open(self.input_file, 'r', encoding=self.encoding) as f:
            lines = f.readlines()
        
        self.stats["lines_processed"] = len(lines)
        print(f"[INFO] Read {len(lines)} lines from file")
        
        # Prepare chunks
        chunks = self._prepare_chunks(lines, chunk_size)
        total_chunks = len(chunks)
        print(f"[INFO] Created {total_chunks} chunks for parallel processing\n")
        
        # Process chunks in parallel
        start_time = time.time()
        
        # Create progress bar if available
        if TQDM_AVAILABLE:
            pbar = async_tqdm(total=total_chunks, desc="Processing chunks", unit="chunk")
        else:
            pbar = None
            print("[INFO] Processing chunks in parallel...")
        
        # Create tasks for all chunks
        tasks = [self._process_chunk(chunk, pbar) for chunk in chunks]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        if pbar:
            pbar.close()
        
        # Check for interruption
        if self._interrupted:
            print("\n[WARNING] Processing was interrupted.")
            return {
                "lines_processed": self.stats["lines_processed"],
                "records_created": 0,
                "api_calls": self.stats["api_calls"],
                "interrupted": True
            }
        
        # Filter out exceptions and collect successful results
        chunk_results = []
        for result in results:
            if isinstance(result, Exception):
                logging.error(f"Task failed with exception: {result}")
            else:
                chunk_results.append(result)
        
        # Deduplicate records from overlapping chunks
        print("\n[PROCESSING] Deduplicating records from overlapping chunks...")
        deduplicated_records = self._deduplicate_records(chunk_results)
        print(f"   [OK] Deduplicated to {len(deduplicated_records)} unique records")
        
        # Post-process to merge updates
        print("\n[PROCESSING] Post-processing records...")
        final_records = self._post_process_records(deduplicated_records)
        self.stats["records_created"] = len(final_records)
        print(f"   [OK] Final count: {len(final_records)} records")
        
        # Save final output
        print(f"\n[SAVING] Saving final output...")
        with open(self.output_file, 'w', encoding=self.encoding) as f:
            json.dump(final_records, f, ensure_ascii=False, indent=2)
        print(f"   [OK] Output saved to '{self.output_file}'")
        
        # Clean up intermediate files
        if self.save_intermediate:
            print("\n[CLEANUP] Cleaning up intermediate files...")
            for chunk_file in self.output_file.parent.glob(f"{self.output_file.stem}_chunk_*.json"):
                try:
                    chunk_file.unlink()
                except:
                    pass
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Print final statistics
        print(f"\n[COMPLETE] Parallel parsing complete!")
        print(f"\n[STATISTICS]")
        print(f"  - Processing time: {processing_time:.1f} seconds")
        print(f"  - Lines processed: {self.stats['lines_processed']:,}")
        print(f"  - Chunks processed: {self.stats['chunks_processed']}/{total_chunks}")
        print(f"  - Records created: {self.stats['records_created']:,}")
        print(f"  - API calls: {self.stats['api_calls']}")
        print(f"  - Average time per chunk: {processing_time/total_chunks:.1f}s")
        
        if self.stats['chunks_failed'] > 0:
            print(f"  - Failed chunks: {self.stats['chunks_failed']}")
            error_file = self.output_file.with_suffix('.errors.txt')
            with open(error_file, 'w', encoding=self.encoding) as f:
                for error in self.stats["errors"]:
                    f.write(f"- {error}\n")
            print(f"  - Errors saved to: {error_file}")
        
        return self.stats

    def parse_genealogy(self, chunk_size: int = 20) -> Dict:
        """
        Synchronous wrapper for the async parse method.
        
        This allows the parser to be used in non-async contexts.
        """
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the async method
            return loop.run_until_complete(
                self.parse_genealogy_async(chunk_size)
            )
        finally:
            loop.close()


# Example usage
if __name__ == "__main__":
    import sys
    
    # Example configuration
    parser = ParallelGenealogyParser(
        input_file="data/intermediate/cleaned_text.txt",
        output_file="data/output/genealogy_data_parallel.json",
        provider="litellm",
        model_name="gpt-4o",
        max_workers=5,
        context_size=5,  # Standardized with parser_v3
        chunk_size=20
    )
    
    # Run parallel parsing
    stats = parser.parse_genealogy(chunk_size=20)
    
    print(f"\nProcessing complete with {stats['records_created']} records")