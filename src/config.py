#!/usr/bin/env python3
"""
config.py

Configuration management for the genealogy processor.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Try to load dotenv if available
try:
    from dotenv import load_dotenv
    # Look for .env file in multiple locations
    # First try the project root (where run_pipeline.py is)
    project_root = Path(__file__).parent.parent
    env_paths = [
        project_root / '.env',  # Project root
        Path('.env'),           # Current directory
        Path.cwd() / '.env'     # Explicit current directory
    ]
    
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path, override=True)
            break
except ImportError:
    pass  # dotenv not installed, skip


class Config:
    """Configuration manager for the genealogy processor"""
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = Path(config_file) if config_file else self._find_config_file()
        self.config = self._load_config()
        self._apply_environment_overrides()
        
    def _find_config_file(self) -> Path:
        """Find configuration file in common locations"""
        # Search paths in order of priority
        search_paths = [
            Path("config/config.yaml"),
            Path("config.yaml"),
            Path("../config/config.yaml"),
            Path.home() / ".genealogy" / "config.yaml"
        ]
        
        for path in search_paths:
            if path.exists():
                return path
                
        # Return default path if none found
        return Path("config/config.yaml")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if not self.config_file.exists():
            # Return default configuration
            return self._get_default_config()
            
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config or {}
        except Exception as e:
            print(f"Warning: Failed to load config from {self.config_file}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'pipeline': {
                'input_dir': 'data/input',
                'output_dir': 'data/output',
                'intermediate_dir': 'data/intermediate',
                'encoding': 'utf-8'
            },
            'merger': {
                'filename_pattern': r'_Page_(\d+)_([a-z]+)_.*?col(\d+)',
                'output_file': 'merged_text.txt'
            },
            'cleaner': {
                'noise_threshold': 0.5,
                'keywords_file': None,
                'output_file': 'cleaned_text.txt',
                'save_stats': True
            },
            'parser': {
                'provider': 'litellm',
                'model': 'gpt-4o-mini',
                'temperature': 0.2,
                'use_function_calling': True,
                'max_retries': 3,
                'retry_delay': 5,
                'output_file': 'genealogy_data.json',
                'save_intermediate': False
            },
            'logging': {
                'level': 'INFO',
                'log_dir': 'logs',
                'console': True,
                'file': True,
                'colored': True
            }
        }
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides"""
        # Map of environment variables to config paths
        # Only allow environment overrides for:
        # - Secrets (API keys)
        # - Runtime settings (log level, directories for CI/CD)
        # All other configuration should be in config.yaml
        env_mappings = {
            # Runtime overrides (useful for CI/CD and debugging)
            'GENEALOGY_INPUT_DIR': ('pipeline', 'input_dir'),
            'GENEALOGY_OUTPUT_DIR': ('pipeline', 'output_dir'),
            'GENEALOGY_LOG_LEVEL': ('logging', 'level'),
            
            # API keys (secrets that should never be in config files)
            'OPENAI_API_KEY': ('parser', 'api_key'),
            'GEMINI_API_KEY': ('parser', 'gemini_api_key'),
            'GOOGLE_API_KEY': ('parser', 'google_api_key'),
            
            # Configuration settings removed - use config.yaml instead:
            # 'GENEALOGY_PROVIDER': ('parser', 'provider'),
            # 'GENEALOGY_MODEL': ('parser', 'model'),
            # 'GENEALOGY_TEMPERATURE': ('parser', 'temperature'),
            # 'GENEALOGY_USE_PARALLEL': ('parser', 'use_parallel'),
            # 'GENEALOGY_MAX_WORKERS': ('parser', 'max_workers'),
            # 'GENEALOGY_CONTEXT_SIZE': ('parser', 'context_size')
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                self._set_nested(self.config, config_path, value)
    
    def _set_nested(self, d: Dict, path: tuple, value: Any):
        """Set nested dictionary value"""
        keys = list(path)
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        
        # Convert string values to appropriate types
        if isinstance(value, str):
            if value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            elif '.' in value and value.replace('.', '').isdigit():
                value = float(value)
                
        d[keys[-1]] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Dot-separated configuration key (e.g., 'pipeline.input_dir')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get pipeline configuration"""
        return self.config.get('pipeline', {})
    
    def get_merger_config(self) -> Dict[str, Any]:
        """Get file merger configuration"""
        config = self.config.get('merger', {})
        
        # Add computed paths
        intermediate_dir = Path(self.get('pipeline.intermediate_dir', 'data/intermediate'))
        config['output_path'] = intermediate_dir / config.get('output_file', 'merged_text.txt')
        
        return config
    
    def get_cleaner_config(self) -> Dict[str, Any]:
        """Get text cleaner configuration"""
        config = self.config.get('cleaner', {})
        
        # Add computed paths
        intermediate_dir = Path(self.get('pipeline.intermediate_dir', 'data/intermediate'))
        config['input_path'] = intermediate_dir / self.get('merger.output_file', 'merged_text.txt')
        config['output_path'] = intermediate_dir / config.get('output_file', 'cleaned_text.txt')
        
        # Load keywords if file specified
        if config.get('keywords_file'):
            keywords_path = Path(config['keywords_file'])
            if keywords_path.exists():
                config['keywords_path'] = keywords_path
                
        return config
    
    def get_parser_config(self) -> Dict[str, Any]:
        """Get genealogy parser configuration"""
        config = self.config.get('parser', {})
        
        # Add computed paths
        intermediate_dir = Path(self.get('pipeline.intermediate_dir', 'data/intermediate'))
        output_dir = Path(self.get('pipeline.output_dir', 'data/output'))
        
        # Check if cleaned text exists, otherwise use merged text
        cleaned_path = intermediate_dir / self.get('cleaner.output_file', 'cleaned_text.txt')
        merged_path = intermediate_dir / self.get('merger.output_file', 'merged_text.txt')
        
        if cleaned_path.exists():
            config['input_path'] = cleaned_path
        else:
            config['input_path'] = merged_path
        
        # Generate output filename based on model if enabled
        base_output = config.get('output_file', 'genealogy_data.json')
        if config.get('use_model_specific_output', False):
            provider = config.get('provider', 'unknown')
            model = config.get('model', 'unknown')
            # Clean model name for filename (replace problematic characters)
            model_clean = model.replace('/', '_').replace(':', '_').replace(' ', '_')
            # Create new filename: genealogy_data_provider_model.json
            base_name = Path(base_output).stem
            extension = Path(base_output).suffix
            output_file = f"{base_name}_{provider}_{model_clean}{extension}"
        else:
            output_file = base_output
            
        config['output_path'] = output_dir / output_file
        
        return config
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.config.get('logging', {})
    
    def save(self, config_file: Optional[Union[str, Path]] = None):
        """
        Save current configuration to file.
        
        Args:
            config_file: Path to save configuration (uses current file if not specified)
        """
        save_path = Path(config_file) if config_file else self.config_file
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return yaml.dump(self.config, default_flow_style=False, allow_unicode=True)


# Global configuration instance
_config = None


def get_config(config_file: Optional[Union[str, Path]] = None) -> Config:
    """
    Get global configuration instance.
    
    Args:
        config_file: Optional configuration file path
        
    Returns:
        Configuration instance
    """
    global _config
    
    if _config is None or config_file is not None:
        _config = Config(config_file)
        
    return _config