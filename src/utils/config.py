"""Configuration management utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration loader and accessor."""
    
    def __init__(self, config_path: str = "configs/jepa_config.yaml"):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'model.encoder.embed_dim')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return self.get(key) is not None

    @property
    def data(self) -> Dict[str, Any]:
        """Return full configuration dictionary."""
        return self.config


def load_config(path: str) -> Config:
    """
    Load configuration from file.
    
    Args:
        path: Path to configuration file
        
    Returns:
        Config object
    """
    return Config(path)
