"""
Model configuration registry for FuzzyShell embedding models.

This module defines available embedding models and their configuration,
allowing for clean model switching and future extensibility.
"""

import os
from typing import Dict, List, Optional

# Registry of available embedding models
EMBEDDING_MODELS = {
    # Default production model - our star performer
    "terminal-minilm-l6": {
        "repo": "Mitchins/minilm-l6-v2-terminal-describer-embeddings-ONNX",
        "description": "MiniLM L6 v2 (384D) - Optimized for terminal commands",
        "files": {
            "model_path": "onnx/int8/model_quantized.onnx",
            "tokenizer_type": "bert",
            "dimensions": 384,
            "model_size_mb": 23
        }
    },
    
    # Comparison baseline (use FUZZYSHELL_MODEL=stock-minilm-l6)
    "stock-minilm-l6": {
        "repo": "onnx-models/all-MiniLM-L6-v2-onnx",
        "description": "Stock MiniLM L6 v2 (384D) - Original model for comparison",
        "files": {
            "model_path": "model.onnx",
            "tokenizer_type": "stock-bert",
            "dimensions": 384,
            "model_size_mb": 90
        },
        "advanced": True
    }
}

# Tokenizer file requirements by type
TOKENIZER_FILES = {
    "bert": [
        "tokenizer.json",
        "config.json",
        "special_tokens_map.json", 
        "tokenizer_config.json",
        "vocab.txt"
    ],
    "stock-bert": [
        "tokenizer.json",
        "config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.txt"
    ]
}

# Default model selection - our star performer
DEFAULT_MODEL = "terminal-minilm-l6"


def get_active_model_key() -> str:
    """
    Get the currently active model key from environment or default.
    
    Returns:
        str: Model key to use
    """
    return os.getenv('FUZZYSHELL_MODEL', DEFAULT_MODEL)


def get_model_config(model_key: Optional[str] = None) -> Dict:
    """
    Get configuration for a specific model.
    
    Args:
        model_key: Model identifier. If None, uses active model.
        
    Returns:
        Dict: Model configuration
        
    Raises:
        ValueError: If model_key is not found in registry
    """
    if model_key is None:
        model_key = get_active_model_key()
    
    if model_key not in EMBEDDING_MODELS:
        available = list(EMBEDDING_MODELS.keys())
        raise ValueError(f"Unknown model '{model_key}'. Available models: {available}")
    
    return EMBEDDING_MODELS[model_key]


def get_required_files(model_key: Optional[str] = None) -> List[str]:
    """
    Get list of all files required for a model.
    
    Args:
        model_key: Model identifier. If None, uses active model.
        
    Returns:
        List[str]: List of required filenames
    """
    config = get_model_config(model_key)
    tokenizer_type = config["files"]["tokenizer_type"]
    
    # Model file + tokenizer files
    files = [os.path.basename(config["files"]["model_path"])]
    files.extend(TOKENIZER_FILES[tokenizer_type])
    
    return files


def list_available_models() -> Dict[str, str]:
    """
    Get a dictionary of available models with descriptions.
    
    Returns:
        Dict[str, str]: Mapping of model_key -> description
    """
    return {
        key: config["description"] 
        for key, config in EMBEDDING_MODELS.items()
    }


def validate_model_config(model_key: str) -> bool:
    """
    Validate that a model configuration is complete and correct.
    
    Args:
        model_key: Model identifier to validate
        
    Returns:
        bool: True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    if model_key not in EMBEDDING_MODELS:
        raise ValueError(f"Model '{model_key}' not found in registry")
    
    config = EMBEDDING_MODELS[model_key]
    
    # Check required top-level keys
    required_keys = ["repo", "description", "files"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Model '{model_key}' missing required key: {key}")
    
    # Check required file configuration
    files_config = config["files"]
    required_file_keys = ["model_path", "tokenizer_type", "dimensions"]
    for key in required_file_keys:
        if key not in files_config:
            raise ValueError(f"Model '{model_key}' missing required file config: {key}")
    
    # Check tokenizer type is supported
    tokenizer_type = files_config["tokenizer_type"]
    if tokenizer_type not in TOKENIZER_FILES:
        available_types = list(TOKENIZER_FILES.keys())
        raise ValueError(f"Model '{model_key}' has unsupported tokenizer type '{tokenizer_type}'. Available: {available_types}")
    
    return True


# Validate all models on import
for model_key in EMBEDDING_MODELS:
    validate_model_config(model_key)