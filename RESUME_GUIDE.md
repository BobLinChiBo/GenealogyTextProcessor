# How to Resume Processing

## Current Status
Your processing stopped at chunk 2 out of 90 due to Gemini API token limits. The checkpoint has been saved.

## Configuration Updated
The `config.yaml` has been updated with smaller chunk sizes:
- `chunk_size`: reduced from 10 to 5 lines
- `context_size`: reduced from 10 to 5 lines

This should help avoid token limit issues with Gemini.

## To Resume Processing

### Option 1: Resume with Current Settings
```bash
python run_pipeline.py --resume
```
This will:
- Continue from chunk 2
- Use the new smaller chunk size (5 lines)
- Keep your existing 7 records

### Option 2: Skip Earlier Stages and Resume
```bash
python run_pipeline.py --skip merge clean --resume
```
This will:
- Skip the merge and clean stages (already done)
- Resume parsing from chunk 2

### Option 3: Start Fresh with New Settings
If you want to start over with the new chunk size:
1. Delete the checkpoint files:
   ```bash
   rm data/output/checkpoints/checkpoint_*.json
   rm data/output/checkpoints/records_*.json
   ```
2. Run the pipeline:
   ```bash
   python run_pipeline.py --skip merge clean
   ```

## Monitor Progress
When running, you'll see:
- Progress bar showing chunks processed
- Real-time record count
- Clear error messages if token limits are hit

## If Token Limits Persist
If you still get token limit errors with chunk_size=5:

1. Further reduce chunk size in `config.yaml`:
   ```yaml
   chunk_size: 3  # Even smaller chunks
   ```

2. Or switch to a different model:
   ```yaml
   provider: "litellm"
   model: "gpt-4o-mini"  # OpenAI model with better token handling
   ```
   
   Or:
   ```yaml
   provider: "gemini"
   model: "gemini-2.5-pro"  # Larger Gemini model with higher limits
   ```

## Cancellation
You can safely press `Ctrl+C` at any time to stop processing. The checkpoint will be saved automatically, and you can resume later.

## Expected Output
Once complete, you'll have:
- `data/output/genealogy_data_gemini_gemini-2.5-flash.json` - Final consolidated output
- Progress saved at each chunk
- Clear statistics showing total records extracted