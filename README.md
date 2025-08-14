# Genealogy Text Processor

An advanced AI-powered pipeline for processing Chinese genealogy texts from OCR output into structured JSON data. This system uses state-of-the-art language models to extract genealogical information from historical Chinese family records.

## Features

- **Three-Stage Processing Pipeline**:
  1. **File Merging**: Intelligently combines individual OCR text files maintaining proper reading order
  2. **Text Cleaning**: Removes OCR noise and filters invalid content
  3. **AI Parsing**: Extracts structured genealogy data using LLMs

- **Multi-Provider LLM Support**:
  - OpenAI (GPT-4, GPT-5 models)
  - Google Gemini (2.5 Flash/Pro series)
  - LiteLLM (unified interface for multiple providers)

- **Advanced Processing Capabilities**:
  - Sequential and parallel processing modes
  - Automatic checkpoint/resume functionality
  - Rate limiting and automatic retry mechanisms
  - Context-aware parsing with configurable window sizes
  - Function calling for structured data extraction

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/GenealogyTextProcessor.git
cd GenealogyTextProcessor
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API keys using `.env` file (recommended) or environment variables:

### Option A: Using .env file (Recommended)
Create a `.env` file in the project root:
```bash
# Copy the example file
cp .env.example .env

# Edit .env with your API keys
```

Example `.env` file:
```bash
# OpenAI API Configuration
OPENAI_API_KEY=your-openai-api-key-here

# Gemini API Configuration  
GEMINI_API_KEY=your-gemini-api-key-here

# Optional: Override provider/model (usually set in config.yaml)
# GENEALOGY_PROVIDER=openai      # Choose: openai, gemini, litellm
# GENEALOGY_MODEL=gpt-4o-mini    # Model to use
```

### Option B: Using environment variables
```bash
# For OpenAI
export OPENAI_API_KEY="your-api-key"

# For Google Gemini
export GEMINI_API_KEY="your-api-key"
# or
export GOOGLE_API_KEY="your-api-key"

# For other providers via LiteLLM
export ANTHROPIC_API_KEY="your-api-key"  # For Claude
```

## Configuration

Edit `config/config.yaml` to customize the pipeline:

```yaml
pipeline:
  input_dir: "data/input"           # OCR text files location
  output_dir: "data/output"         # Final JSON output
  
parser:
  provider: "openai"                # Choose: openai, gemini, litellm
  model: "gpt-4o-mini"              # Select specific model
  temperature: 0.2                  # Lower = more deterministic
  use_parallel: false               # Enable parallel processing
  max_workers: 3                    # Concurrent API calls (parallel mode)
```

## Usage

### Basic Usage

Run the complete pipeline:
```bash
python run_pipeline.py
```

### Advanced Options

```bash
# Skip stages that have already been completed
python run_pipeline.py --skip merge clean

# Resume from checkpoint (if interrupted)
python run_pipeline.py --resume

# Use parallel processing for faster parsing
python run_pipeline.py --parallel

# Use custom configuration
python run_pipeline.py -c custom_config.yaml

# Override directories
python run_pipeline.py --input-dir /path/to/input --output-dir /path/to/output
```

### Command-Line Options

- `-c, --config`: Path to custom configuration file
- `--skip [merge|clean|parse]`: Skip specified pipeline stages
- `--resume`: Resume parsing from last checkpoint
- `--parallel`: Enable parallel processing mode
- `--input-dir`: Override input directory
- `--output-dir`: Override output directory
- `-v, --verbose`: Enable verbose logging

## Input Format

The pipeline expects OCR text files with specific naming patterns:

- **Multi-column format**: `Wang2017_Page_001_left_col01_final_deskewed.txt`
- **Side-only format**: `Doc_Page_001_left_enhanced.txt`
- **Simple page format**: `Shu_Page_087_enhanced.txt`

Files are automatically sorted and merged in the correct reading order.

## Output Format

The pipeline produces structured JSON following the genealogy schema:

```json
{
  "records": [
    {
      "name": "王大明",
      "sex": "male",
      "father": "王父",
      "birth_order": 1,
      "courtesy": "子明",
      "birth_time": "明嘉靖二十年辛丑十月十二日寅時",
      "death_time": "萬歷二十七年己亥八月十九日午時",
      "children": [
        {"order": 1, "name": "王小明", "sex": "male"}
      ],
      "info": "Biographical details...",
      "original_text": "Original OCR text line",
      "note": "Parsing reasoning",
      "is_update_for_previous": false,
      "skip": false
    }
  ]
}
```

## Pipeline Stages

### 1. File Merger
- Combines individual OCR files in proper reading order
- Supports multiple filename patterns
- Handles multi-column layouts intelligently

### 2. Text Cleaner
- Removes OCR noise and artifacts
- Filters lines based on noise threshold
- Preserves valid Chinese genealogy content

### 3. AI Parser
- Uses LLMs to extract structured data
- Maintains context across chunks
- Supports both sequential and parallel processing
- Automatic checkpoint saving for interruption recovery

## Performance Optimization

### Parallel Processing
Enable parallel mode for faster processing:
```yaml
parser:
  use_parallel: true
  max_workers: 5                    # Adjust based on API limits
  requests_per_minute: 30           # Rate limiting
  tokens_per_minute: 25000          # Token budget
```

### Rate Limiting
The system automatically handles rate limits:
- Configurable requests/tokens per minute
- Automatic worker adjustment on rate limit errors
- Slow start option to gradually increase parallel workers

## Troubleshooting

### Common Issues

1. **API Rate Limits**: Reduce `max_workers` or enable `slow_start`
2. **Memory Issues**: Reduce `chunk_size` and `context_size`
3. **Interrupted Processing**: Use `--resume` flag to continue
4. **OCR Noise**: Adjust `noise_threshold` in cleaner config

### Logging

Logs are saved to the `logs/` directory with timestamps:
```
logs/genealogy_20250814_143319.log
```

Set log level in config or via environment:
```bash
export GENEALOGY_LOG_LEVEL=DEBUG
```

## Project Structure

```
GenealogyTextProcessor/
├── config/
│   └── config.yaml              # Main configuration
├── data/
│   ├── input/                   # OCR text files
│   ├── intermediate/            # Merged and cleaned files
│   └── output/                  # Final JSON output
├── src/
│   ├── pipeline/                # Core pipeline modules
│   │   ├── file_merger.py
│   │   ├── text_cleaner.py
│   │   ├── genealogy_parser_v3.py
│   │   └── genealogy_parser_parallel.py
│   ├── llm/                     # LLM provider interfaces
│   ├── prompts/                 # System prompts
│   └── utils/                   # Utility functions
├── schemas/
│   └── genealogy.schema.json    # JSON schema definition
├── run_pipeline.py              # Main entry point
└── requirements.txt             # Python dependencies
```

## Requirements

- Python 3.8+
- API key for chosen LLM provider
- 8GB+ RAM recommended for large datasets

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on GitHub.