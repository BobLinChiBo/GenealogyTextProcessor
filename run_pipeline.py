#!/usr/bin/env python3
"""
run_pipeline.py

Main entry point for the Chinese Genealogy Text Processing Pipeline.
Orchestrates the three-stage process: merge, clean, and parse.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import get_config
from utils.logger import setup_logging, get_logger
from utils.file_handler import ensure_directory, format_file_size, get_file_size
from pipeline.file_merger import FileMerger
from pipeline.text_cleaner import TextCleaner
# Use v3 parser with LLM provider support
from pipeline.genealogy_parser_v3 import GenealogyParser
from pipeline.genealogy_parser_parallel import ParallelGenealogyParser


class GenealogyPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config_file: str = None, skip_stages: list = None, resume: bool = False):
        """
        Initialize the pipeline.
        
        Args:
            config_file: Path to configuration file
            skip_stages: List of stages to skip ['merge', 'clean', 'parse']
            resume: Resume from checkpoint for parse stage
        """
        self.config = get_config(config_file)
        self.skip_stages = skip_stages or []
        self.resume = resume
        
        # Setup logging
        log_config = self.config.get_logging_config()
        self.logger = setup_logging(**log_config)
        
        # Track execution time
        self.start_time = None
        
    def run(self):
        """Run the complete pipeline"""
        self.start_time = time.time()
        
        self.logger.info("=" * 60)
        self.logger.info("Chinese Genealogy Text Processing Pipeline")
        self.logger.info("=" * 60)
        self.logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Stage 1: Merge files
            if 'merge' not in self.skip_stages:
                self._run_merge_stage()
            else:
                self.logger.info("Skipping merge stage")
            
            # Stage 2: Clean text
            if 'clean' not in self.skip_stages:
                self._run_clean_stage()
            else:
                self.logger.info("Skipping clean stage")
            
            # Stage 3: Parse genealogy
            if 'parse' not in self.skip_stages:
                self._run_parse_stage()
            else:
                self.logger.info("Skipping parse stage")
            
            # Final summary
            self._print_summary()
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def _run_merge_stage(self):
        """Run file merging stage"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STAGE 1: File Merging")
        self.logger.info("=" * 60)
        
        # Get configuration
        pipeline_config = self.config.get_pipeline_config()
        merger_config = self.config.get_merger_config()
        
        input_dir = Path(pipeline_config['input_dir'])
        output_path = merger_config['output_path']
        
        # Validate input directory
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Create merger and run
        merger = FileMerger(
            input_dir=str(input_dir),
            output_file=str(output_path),
            filename_pattern=merger_config.get('filename_pattern'),
            filename_patterns=merger_config.get('filename_patterns'),
            encoding=pipeline_config.get('encoding', 'utf-8')
        )
        
        self.logger.info(f"Input directory: {input_dir}")
        self.logger.info(f"Output file: {output_path}")
        
        stats = merger.merge_files(show_progress=True)
        
        # Log results
        self.logger.info(f"Merge complete:")
        self.logger.info(f"  - Files processed: {stats['files_processed']}")
        self.logger.info(f"  - Total characters: {stats['total_characters']:,}")
        self.logger.info(f"  - Output size: {format_file_size(get_file_size(output_path))}")
    
    def _run_clean_stage(self):
        """Run text cleaning stage"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STAGE 2: Text Cleaning")
        self.logger.info("=" * 60)
        
        # Get configuration
        pipeline_config = self.config.get_pipeline_config()
        cleaner_config = self.config.get_cleaner_config()
        
        input_path = cleaner_config['input_path']
        output_path = cleaner_config['output_path']
        
        # Validate input file
        if not input_path.exists():
            raise FileNotFoundError(
                f"Input file not found: {input_path}. "
                "Please run merge stage first."
            )
        
        # Create cleaner and run
        cleaner = TextCleaner(
            input_file=str(input_path),
            output_file=str(output_path),
            noise_threshold=cleaner_config.get('noise_threshold', 0.15),
            keywords_file=cleaner_config.get('keywords_path'),
            encoding=pipeline_config.get('encoding', 'utf-8')
        )
        
        self.logger.info(f"Input file: {input_path}")
        self.logger.info(f"Output file: {output_path}")
        self.logger.info(f"Noise threshold: {cleaner_config.get('noise_threshold', 0.15)}")
        
        stats = cleaner.clean_text(save_stats=cleaner_config.get('save_stats', True))
        
        # Log results
        self.logger.info(f"Cleaning complete:")
        self.logger.info(f"  - Lines processed: {stats['total_lines']:,}")
        self.logger.info(f"  - Valid lines: {stats['valid_lines']:,}")
        self.logger.info(f"  - Retention rate: {stats['retention_rate']*100:.1f}%")
        self.logger.info(f"  - Output size: {format_file_size(get_file_size(output_path))}")
    
    def _run_parse_stage(self):
        """Run genealogy parsing stage"""
        print("\n" + "=" * 60)
        print("STAGE 3: Genealogy Parsing (AI)")
        print("=" * 60)
        
        # Get configuration
        pipeline_config = self.config.get_pipeline_config()
        parser_config = self.config.get_parser_config()
        
        input_path = parser_config['input_path']
        output_path = parser_config['output_path']
        
        # Validate input file
        if not input_path.exists():
            raise FileNotFoundError(
                f"Input file not found: {input_path}. "
                "Please run clean stage first."
            )
        
        # Check for existing checkpoint
        input_hash = str(abs(hash(str(input_path))))[:8]
        checkpoint_dir = output_path.parent / "checkpoints"
        checkpoint_file = checkpoint_dir / f"checkpoint_{input_hash}.json"
        
        if checkpoint_file.exists() and not self.resume:
            print(f"\n[WARNING] Found existing checkpoint from previous run")
            print(f"   Checkpoint: {checkpoint_file}")
            print(f"   Use --resume flag to continue from checkpoint")
            response = input("\nDo you want to resume? (y/n): ")
            if response.lower() == 'y':
                self.resume = True
                print("\n[OK] Resuming from checkpoint...")
        
        # Ensure output directory exists
        ensure_directory(output_path.parent)
        
        # Create parser and run with provider support
        # Select the appropriate API key based on provider
        provider = parser_config.get('provider', 'litellm')
        if provider == 'gemini':
            api_key = parser_config.get('gemini_api_key') or parser_config.get('google_api_key')
        else:
            api_key = parser_config.get('api_key')
        
        # Check if parallel processing is enabled
        use_parallel = parser_config.get('use_parallel', False)
        
        # Use different output filename for parallel processing
        if use_parallel:
            # Override output_path if parallel-specific filename is configured
            parallel_output_file = parser_config.get('output_file_parallel')
            if parallel_output_file:
                output_path = output_path.parent / parallel_output_file
        
        if use_parallel:
            print("[INFO] Using PARALLEL processing mode")
            parser = ParallelGenealogyParser(
                input_file=str(input_path),
                output_file=str(output_path),
                provider=provider,
                model_name=parser_config.get('model'),
                api_key=api_key,
                temperature=parser_config.get('temperature', 0.2),
                max_workers=parser_config.get('max_workers', 5),
                context_size=parser_config.get('context_size', 5),  # Standardized parameter
                use_function_calling=parser_config.get('use_function_calling', True),
                max_retries=parser_config.get('max_retries', 3),
                retry_delay=parser_config.get('retry_delay', 5),
                encoding=pipeline_config.get('encoding', 'utf-8'),
                save_intermediate=parser_config.get('save_intermediate', True),
                requests_per_minute=parser_config.get('requests_per_minute', 30),
                auto_adjust_workers=parser_config.get('auto_adjust_workers', True),
                slow_start=parser_config.get('slow_start', True)
            )
        else:
            print("[INFO] Using SEQUENTIAL processing mode")
            parser = GenealogyParser(
                input_file=str(input_path),
                output_file=str(output_path),
                provider=provider,
                model_name=parser_config.get('model'),
                api_key=api_key,
                temperature=parser_config.get('temperature', 0.2),
                use_function_calling=parser_config.get('use_function_calling', True),
                max_retries=parser_config.get('max_retries', 3),
                retry_delay=parser_config.get('retry_delay', 5),
                encoding=pipeline_config.get('encoding', 'utf-8'),
                resume=self.resume,
                save_intermediate=parser_config.get('save_intermediate', True),
                merge_updates=parser_config.get('merge_updates', True)
            )
        
        print(f"\n[INPUT] {input_path}")
        print(f"[OUTPUT] {output_path}")
        print(f"[PROVIDER] {provider}")
        print(f"[MODEL] {parser_config.get('model')}")
        print(f"[TEMPERATURE] {parser_config.get('temperature', 0.2)}")
        if use_parallel:
            print(f"[MAX WORKERS] {parser_config.get('max_workers', 3)}")
            print(f"[CONTEXT SIZE] {parser_config.get('context_size', 5)} lines")
            print(f"[RATE LIMIT] {parser_config.get('requests_per_minute', 30)} req/min")
            print(f"[TOKEN LIMIT] {parser_config.get('tokens_per_minute', 25000)} tokens/min")
            print(f"[AUTO ADJUST] {parser_config.get('auto_adjust_workers', True)}")
            print(f"[SLOW START] {parser_config.get('slow_start', True)}")
        print()
        
        try:
            chunk_size = parser_config.get('chunk_size', 20)
            context_size = parser_config.get('context_size', 10)
            
            # Both parsers now use the same context_size parameter
            stats = parser.parse_genealogy(chunk_size=chunk_size, context_size=context_size)
        except KeyboardInterrupt:
            print("\n\n[WARNING] Pipeline interrupted by user")
            print("[INFO] Progress has been saved. Use --resume to continue.")
            return
        except Exception as parse_error:
            self.logger.error(f"Parse error details: {parse_error}")
            self.logger.error(f"Error type: {type(parse_error).__name__}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        # Log results
        if stats.get('interrupted'):
            print("\n[WARNING] Processing was interrupted.")
            print("[INFO] Progress saved. Run with --resume to continue.")
        elif stats.get('checkpoint_saved'):
            print("\n[WARNING] Parsing paused due to API limits.")
            print("[INFO] Progress saved. Run with --resume to continue.")
        else:
            print("\n[OK] Parsing complete!")
        
        print(f"\n[STATISTICS]")
        print(f"  - Lines processed: {stats['lines_processed']:,}")
        print(f"  - Records created: {stats['records_created']:,}")
        print(f"  - API calls: {stats['api_calls']}")
        
        if output_path.exists():
            print(f"  - Output size: {format_file_size(get_file_size(output_path))}")
    
    def _print_summary(self):
        """Print final summary"""
        execution_time = time.time() - self.start_time
        
        print("\n" + "=" * 60)
        print("[OK] PIPELINE COMPLETE")
        print("=" * 60)
        print(f"[TIME] Total execution time: {execution_time:.1f} seconds")
        print(f"[END TIME] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check for final output
        parser_config = self.config.get_parser_config()
        output_path = parser_config['output_path']
        
        if output_path.exists():
            print(f"\n[FINAL OUTPUT] {output_path}")
            print(f"[OUTPUT SIZE] {format_file_size(get_file_size(output_path))}")
        
        print("\n[SUCCESS] Pipeline completed successfully!")


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Chinese Genealogy Text Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline stages:
  1. Merge: Combine individual character files into merged text
  2. Clean: Filter out OCR noise and invalid lines
  3. Parse: Use AI to extract structured genealogy data

Example usage:
  # Run complete pipeline
  python run_pipeline.py
  
  # Run with custom config
  python run_pipeline.py -c my_config.yaml
  
  # Skip merge stage (if already done)
  python run_pipeline.py --skip merge
  
  # Run only parse stage
  python run_pipeline.py --skip merge clean
        """
    )
    
    parser.add_argument(
        "-c", "--config",
        help="Path to configuration file (default: config/config.yaml)"
    )
    
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=['merge', 'clean', 'parse'],
        help="Skip specified stages"
    )
    
    parser.add_argument(
        "--input-dir",
        help="Override input directory from config"
    )
    
    parser.add_argument(
        "--output-dir",
        help="Override output directory from config"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume genealogy parsing from last checkpoint"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel processing mode for faster parsing"
    )
    
    args = parser.parse_args()
    
    # Apply command-line overrides
    if args.input_dir:
        import os
        os.environ['GENEALOGY_INPUT_DIR'] = args.input_dir
        
    if args.output_dir:
        import os
        os.environ['GENEALOGY_OUTPUT_DIR'] = args.output_dir
        
    if args.verbose:
        import os
        os.environ['GENEALOGY_LOG_LEVEL'] = 'DEBUG'
    
    if args.parallel:
        import os
        os.environ['GENEALOGY_USE_PARALLEL'] = 'true'
    
    try:
        # Create and run pipeline
        pipeline = GenealogyPipeline(
            config_file=args.config,
            skip_stages=args.skip or [],
            resume=args.resume
        )
        pipeline.run()
        
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())