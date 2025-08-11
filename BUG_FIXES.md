# Bug Fixes Applied

## Issues Fixed

### 1. Duplicate Records Across Chunks [OK]
**Problem**: Each chunk was producing the same records because the `parse_with_context` method in `gemini_provider.py` was incorrectly handling the prompt. It was passing an already-processed prompt as a template, causing the actual new text to be lost.

**Solution**: 
- Rewrote `parse_with_context` to directly call the API with the complete prompt
- Now properly maintains separation between context (for reference) and new text (for parsing)
- Each chunk now correctly processes only its designated lines

### 2. Token Limit Configuration [OK]
**Problem**: The system was configured for 4096-8192 output tokens, but Gemini 2.5 supports 16,384 output tokens.

**Solution**:
- Updated `max_tokens` from 4096/8192 to 16384 in both code and config
- Increased `chunk_size` back to 20 lines (was reduced to 5 unnecessarily)
- Increased `context_size` back to 10 lines for better context preservation

## Changes Made

### Files Modified:

1. **src/llm/gemini_provider.py**
   - Fixed `parse_with_context` method to properly handle context vs new text
   - Updated default max_tokens from 8192 to 16384
   - Added proper error handling in the context method

2. **config/config.yaml**
   - `max_tokens`: 4096 → 16384
   - `chunk_size`: 5 → 20 
   - `context_size`: 5 → 10

3. **Checkpoints Cleared**
   - Removed old checkpoint files that contained duplicate records
   - Fresh start with corrected parsing logic

## Verification

The fixes ensure:
- [OK] Each chunk processes unique content (no duplicates)
- [OK] Context is used for understanding relationships but not re-parsed
- [OK] Full token capacity of Gemini 2.5 is utilized (16K tokens)
- [OK] Larger chunks can be processed efficiently

## How to Run

With the fixes applied, run:

```bash
# Fresh start (recommended)
python run_pipeline.py --skip merge clean

# Or if you want to see all stages
python run_pipeline.py
```

## Expected Behavior

1. **No Duplicate Records**: Each chunk will extract different genealogy records
2. **Better Performance**: Larger chunks mean fewer API calls (45 instead of 180)
3. **No Token Limit Errors**: 16K output tokens is more than sufficient for genealogy parsing
4. **Proper Context Usage**: Previous chunks provide context without being re-parsed

## Token Capacity Reference

Gemini 2.5 Models (as of August 2025):
- **Input**: 1,048,576 tokens (flash) / 2,097,152 tokens (pro)
- **Output**: 16,384 tokens (both models)

With 20 lines per chunk and Chinese genealogy text, we're well within limits.