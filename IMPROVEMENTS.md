# Pipeline Improvements

## Overview
The genealogy text processing pipeline has been enhanced with the following improvements for better user experience and control.

## Key Improvements

### 1. Real-time Progress Display
- **No duplicate logging**: Removed duplicate logger instances that were causing confusing output
- **Clear progress indicators**: Uses emoji and formatted output for better visibility
- **Progress bars**: When `tqdm` is installed, displays progress bars for chunk processing
- **Live status updates**: Shows current chunk being processed and records found in real-time

### 2. Graceful Cancellation (Ctrl+C)
- **Signal handling**: Properly handles SIGINT (Ctrl+C) to allow graceful interruption
- **Automatic checkpoint saving**: When interrupted, automatically saves progress
- **Clear messages**: Displays clear instructions on how to resume
- **No data loss**: All processed records up to the interruption point are preserved

### 3. Automatic Resume Detection
- **Checkpoint detection**: Automatically detects existing checkpoints when starting
- **Interactive prompt**: Asks user if they want to resume from checkpoint
- **Progress tracking**: Shows previous progress (records created, API calls made)
- **Seamless continuation**: Picks up exactly where it left off

### 4. Improved Output Management
- **Consolidated final output**: Creates a single final JSON file when all chunks complete
- **Intermediate file cleanup**: Automatically removes intermediate chunk files after successful completion
- **Clear file naming**: Output files use descriptive names including provider and model
- **Error tracking**: Saves errors to a separate `.errors.txt` file for review

### 5. Better Status Display
- **Chunk-by-chunk progress**: Shows which chunk is being processed (e.g., "Chunk 3/10")
- **Line range display**: Shows which lines are being processed in each chunk
- **Context information**: Displays how many context lines are included
- **API call tracking**: Shows total API calls made
- **Record count**: Displays running total of records extracted

## Usage Examples

### Normal Run
```bash
python run_pipeline.py
```

### Resume from Checkpoint
```bash
python run_pipeline.py --resume
```

### Skip Earlier Stages
```bash
python run_pipeline.py --skip merge clean
```

### With Progress Bar (requires tqdm)
```bash
pip install tqdm
python run_pipeline.py
```

### Graceful Interruption
1. Start the pipeline: `python run_pipeline.py`
2. Press `Ctrl+C` during processing
3. Pipeline saves checkpoint and exits gracefully
4. Resume later with: `python run_pipeline.py --resume`

## Visual Indicators

The improved pipeline uses clear visual indicators:
- 📄 Input/output files
- [SAVING] Saving operations
- [OK] Successful operations
- [ERROR] Failures
- ⚠️ Warnings or interruptions
- [PROCESSING] Processing operations
- 📊 Statistics
- 🎉 Completion
- 📦 Chunk processing
- 🤖 AI model information

## Configuration

The improvements work with the existing configuration in `config/config.yaml`:
- `save_intermediate`: Controls whether to save intermediate chunk files
- `chunk_size`: Number of lines to process per API call
- `use_model_specific_output`: Adds model name to output files

## Technical Details

### Files Modified
1. `src/pipeline/genealogy_parser_v3.py`:
   - Added signal handling for Ctrl+C
   - Improved progress display
   - Removed duplicate logging
   - Added tqdm support
   - Better checkpoint management

2. `run_pipeline.py`:
   - Added automatic checkpoint detection
   - Interactive resume prompt
   - Better error handling for interruptions
   - Cleaner status display

### Dependencies
- **Required**: No new required dependencies
- **Optional**: `tqdm` for progress bars (already in requirements.txt)

## Benefits

1. **Better User Experience**: Clear, real-time feedback on processing progress
2. **Resilient Processing**: Can be safely interrupted and resumed
3. **No Lost Work**: All progress is saved in checkpoints
4. **Cleaner Output**: No duplicate logging or confusing messages
5. **Professional Feel**: Modern CLI interface with progress indicators

## Testing

To test the improvements:
```bash
python test_improvements.py
```

This will demonstrate:
- Real-time progress display
- Graceful cancellation (press Ctrl+C during processing)
- Checkpoint detection and resume functionality