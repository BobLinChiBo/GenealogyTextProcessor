# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Chinese Genealogy Text Processing Pipeline that processes OCR text files from genealogy documents. The pipeline consists of three main stages:
1. **Merge**: Combines individual character/column files into a single merged text file
2. **Clean**: Filters out OCR noise and invalid lines based on Chinese character ratio and genealogy keywords
3. **Parse**: Uses AI models (ChatGPT, Gemini, or others via LiteLLM) to extract structured genealogy data from the cleaned text

## Key Commands

### Running the Pipeline
```bash
# Run complete pipeline
python run_pipeline.py

# Run with custom config
python run_pipeline.py -c my_config.yaml

# Skip specific stages
python run_pipeline.py --skip merge  # Skip merge if already done
python run_pipeline.py --skip merge clean  # Run only parse stage

# Run with verbose logging
python run_pipeline.py -v
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_pipeline.py -v

# Run with coverage
pytest --cov=src tests/

# Test improvements (progress bars, graceful cancellation)
python test_improvements.py
```

### Code Quality
```bash
# Format code with black
black src/ tests/

# Check code style with flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/
```

### Installation
```bash
# Install all dependencies
pip install -r requirements.txt

# Install with development dependencies (from setup.py)
pip install -e .[dev]

# Install with enhanced features (progress bars, colors)
pip install -e .[enhanced]
```

## Architecture

### Core Components

The pipeline follows a modular architecture with clear separation of concerns:

- **Pipeline Orchestration** (`run_pipeline.py`): Main entry point that coordinates the three stages
- **Configuration Management** (`src/config.py`): Centralized configuration with YAML support and environment variable overrides
- **Stage Processors** (`src/pipeline/`):
  - `file_merger.py`: Merges OCR column files in proper reading order (right-to-left for Chinese)
  - `text_cleaner.py`: Validates and cleans text based on Chinese character ratio and genealogy keywords
  - `genealogy_parser_v3.py`: Uses modular LLM providers to extract structured genealogy records
- **LLM Providers** (`src/llm/`):
  - `base.py`: Abstract base class defining the LLM provider interface
  - `openai_provider.py`: OpenAI ChatGPT implementation
  - `gemini_provider.py`: Google Gemini implementation
  - `litellm_provider.py`: Unified provider supporting 100+ models
- **Utilities** (`src/utils/`):
  - `chinese_detector.py`: Detects Chinese characters and genealogy-specific keywords
  - `file_handler.py`: File I/O operations and directory management
  - `logger.py`: Logging configuration with colored output support

### Data Flow

1. **Input**: Individual OCR text files in `data/input/` with naming pattern: `PREFIX_Page_XXX_SIDE_border_colYY_ID.txt`
2. **Intermediate**: Processed files stored in `data/intermediate/`:
   - `merged_text.txt`: Combined text from all input files
   - `cleaned_text.txt`: Filtered text with valid genealogy content
3. **Output**: Final structured data in `data/output/genealogy_data.json`

### File Processing Order

The merger processes files in traditional Chinese reading order:
- Within each page: right side → left side
- Within each side: columns from right to left (highest number first)
- Pages in sequential order

## Configuration

Configuration is managed through `config/config.yaml` with environment variable overrides:

### Environment Variables (Runtime Overrides Only)
- `GENEALOGY_INPUT_DIR`: Override input directory (useful for CI/CD)
- `GENEALOGY_OUTPUT_DIR`: Override output directory (useful for CI/CD)
- `GENEALOGY_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

### API Keys (Store in .env file)
- `OPENAI_API_KEY`: API key for OpenAI/LiteLLM
- `GEMINI_API_KEY` or `GOOGLE_API_KEY`: API key for Google Gemini
- `ANTHROPIC_API_KEY`: API key for Claude via LiteLLM

**Note**: Model selection is configured exclusively through `config/config.yaml`. The `.env` file should only contain API keys and other secrets, not configuration settings.

### Key Configuration Parameters (in config.yaml)
- `merger.filename_pattern`: Regex for parsing input filenames
- `cleaner.noise_threshold`: Maximum ratio of non-Chinese characters (0-1)
- `parser.provider`: LLM provider to use (openai, gemini, litellm)
- `parser.model`: Model name (e.g., gpt-5, gpt-5-mini, gpt-5-nano, gpt-4o, gemini-2.5-flash)
- `parser.temperature`: Model temperature for controlling randomness (0.0-2.0)
- `parser.use_function_calling`: Enable structured output via function calling
- `parser.max_retries`: Retry attempts for failed API calls

## Important Notes

- The project requires an API key for your chosen LLM provider (OpenAI, Gemini, etc.)
- **LiteLLM** is recommended as it provides a unified interface for multiple providers
- Input files must follow the specific naming convention for proper ordering
- The cleaner uses genealogy-specific keywords to identify valid content
- All text processing assumes UTF-8 encoding by default
- The pipeline supports checkpointing and resuming for interrupted parsing sessions
- Progress is displayed with real-time updates and optional progress bars (if tqdm is installed)

## Supported LLM Models

### OpenAI (via `openai` or `litellm` provider)
- `gpt-5` (recommended - PhD-level expertise, 45% fewer errors than GPT-4o)
- `gpt-5-mini` (lightweight version for cost-sensitive applications)
- `gpt-5-nano` (optimized for ultra-low latency)
- `gpt-5-chat` (built for advanced, natural conversations)
- `gpt-4o`, `gpt-4o-mini` (previous generation)
- `gpt-3.5-turbo` (legacy model)

### Google Gemini (via `gemini` or `litellm` provider)
- `gemini-2.5-flash` (recommended - best price/performance, includes thinking capabilities)
- `gemini-2.5-pro` (advanced model with thinking)
- `gemini-2.5-flash-lite` (fastest and lowest cost - $0.10/$0.40 per 1M tokens)

### Via LiteLLM (100+ models)
- Anthropic Claude: `claude-3-opus-20240229`, `claude-3-sonnet`
- Mistral: `mistral-large`, `mistral-medium`
- And many more - see [LiteLLM docs](https://docs.litellm.ai/docs/providers)

## Development Workflow

### Setting Up Development Environment
```bash
# Clone the repository
git clone <repository-url>
cd genealogy-text-processor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .[dev,enhanced]

# Create .env file for API keys ONLY (not for configuration)
echo "OPENAI_API_KEY=your-key-here" > .env
# or
echo "GEMINI_API_KEY=your-key-here" > .env

# To change models, edit config/config.yaml, NOT the .env file
```

### Running Tests Before Committing
```bash
# Format code
black src/ tests/

# Check code style
flake8 src/ tests/

# Run tests
pytest tests/ -v

# Type checking (optional)
mypy src/
```

### Debugging Tips

1. **Enable verbose logging**:
   ```bash
   python run_pipeline.py -v
   # or
   export GENEALOGY_LOG_LEVEL=DEBUG
   ```

2. **Test individual stages**:
   ```bash
   # Test only merge
   python run_pipeline.py --skip clean parse
   
   # Test only clean
   python run_pipeline.py --skip merge parse
   
   # Test only parse
   python run_pipeline.py --skip merge clean
   ```

3. **Check intermediate files**:
   - `data/intermediate/merged_text.txt` - Output from merge stage
   - `data/intermediate/cleaned_text.txt` - Output from clean stage
   - `data/output/checkpoints/` - Saved progress from interrupted parsing

4. **Resume interrupted parsing**:
   ```bash
   python run_pipeline.py --skip merge clean --resume
   ```