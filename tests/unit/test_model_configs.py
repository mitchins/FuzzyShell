"""
Tests for model configuration system.
"""

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock

# The src path is already configured in pyproject.toml

from fuzzyshell.model_configs import (
    get_active_model_key, get_model_config, get_required_files,
    list_available_models, validate_model_config, EMBEDDING_MODELS,
    TOKENIZER_FILES, DEFAULT_MODEL
)


class TestModelConfigs:
    """Test model configuration functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        # Clear any environment variables
        if 'FUZZYSHELL_MODEL' in os.environ:
            del os.environ['FUZZYSHELL_MODEL']
    
    def test_default_model_selection(self):
        """Test that default model is selected when no environment variable is set."""
        model_key = get_active_model_key()
        assert model_key == DEFAULT_MODEL
        assert model_key == "terminal-minilm-l6"
    
    def test_environment_model_selection(self):
        """Test model selection from environment variable."""
        with patch.dict(os.environ, {'FUZZYSHELL_MODEL': 'terminal-minilm-l6'}):
            model_key = get_active_model_key()
            assert model_key == "terminal-minilm-l6"
    
    def test_get_model_config_valid(self):
        """Test getting configuration for valid models."""
        # Test terminal-minilm-l6
        config = get_model_config("terminal-minilm-l6")
        assert "repo" in config
        assert "description" in config
        assert "files" in config
        
        files_config = config["files"]
        assert "model_path" in files_config
        assert "tokenizer_type" in files_config
        assert "dimensions" in files_config
        
        # Test terminal-minilm-l6
        config = get_model_config("terminal-minilm-l6")
        assert config["files"]["tokenizer_type"] == "bert"
        assert config["files"]["dimensions"] == 384
    
    def test_get_model_config_invalid(self):
        """Test getting configuration for invalid model."""
        with pytest.raises(ValueError) as exc_info:
            get_model_config("nonexistent-model")
        
        assert "Unknown model" in str(exc_info.value)
        assert "nonexistent-model" in str(exc_info.value)
    
    def test_get_model_config_none_uses_active(self):
        """Test that get_model_config with None uses active model."""
        with patch.dict(os.environ, {'FUZZYSHELL_MODEL': 'terminal-minilm-l6'}):
            config = get_model_config(None)
            assert config["files"]["tokenizer_type"] == "bert"
    
    def test_get_required_files(self):
        """Test getting required files for models."""
        # Test XLM-RoBERTa model files
        files = get_required_files("terminal-minilm-l6")
        assert "model_quantized.onnx" in files
        assert "tokenizer.json" in files
        assert "config.json" in files
        assert "special_tokens_map.json" in files
        assert "tokenizer_config.json" in files
        assert "vocab.txt" in files
        
        # Test BERT model files
        files = get_required_files("terminal-minilm-l6")
        assert "model_quantized.onnx" in files
        assert "tokenizer.json" in files
        assert "config.json" in files
        assert "special_tokens_map.json" in files
        assert "tokenizer_config.json" in files
        assert "vocab.txt" in files  # BERT-specific
        assert "sentencepiece.bpe.model" not in files  # Not for BERT
    
    def test_list_available_models(self):
        """Test listing available models."""
        models = list_available_models()
        assert isinstance(models, dict)
        assert "terminal-minilm-l6" in models
        assert "terminal-minilm-l6" in models
        
        # Check descriptions are present
        for model_key, description in models.items():
            assert isinstance(description, str)
            assert len(description) > 0
    
    def test_validate_model_config_valid(self):
        """Test validation of valid model configurations."""
        for model_key in EMBEDDING_MODELS:
            assert validate_model_config(model_key) is True
    
    def test_validate_model_config_invalid(self):
        """Test validation of invalid model configurations."""
        with pytest.raises(ValueError):
            validate_model_config("nonexistent-model")
    
    def test_tokenizer_files_consistency(self):
        """Test that all tokenizer types referenced in models exist in TOKENIZER_FILES."""
        for model_key, config in EMBEDDING_MODELS.items():
            tokenizer_type = config["files"]["tokenizer_type"]
            assert tokenizer_type in TOKENIZER_FILES, \
                f"Model {model_key} references unknown tokenizer type: {tokenizer_type}"
    
    def test_model_config_structure(self):
        """Test that all models have consistent configuration structure."""
        required_top_keys = ["repo", "description", "files"]
        required_file_keys = ["model_path", "tokenizer_type", "dimensions"]
        
        for model_key, config in EMBEDDING_MODELS.items():
            # Check top-level keys
            for key in required_top_keys:
                assert key in config, f"Model {model_key} missing key: {key}"
            
            # Check files configuration
            files_config = config["files"]
            for key in required_file_keys:
                assert key in files_config, f"Model {model_key} files config missing key: {key}"
            
            # Check types
            assert isinstance(config["repo"], str)
            assert isinstance(config["description"], str)
            assert isinstance(files_config["model_path"], str)
            assert isinstance(files_config["tokenizer_type"], str)
            assert isinstance(files_config["dimensions"], int)
    
    def test_repo_format(self):
        """Test that repository names follow HuggingFace format."""
        import re
        for model_key, config in EMBEDDING_MODELS.items():
            repo = config["repo"]
            assert re.match(r'^[^/]+/[^/]+$', repo), \
                f"Model {model_key} has invalid repo format: {repo}"
    
    def test_dimensions_consistency(self):
        """Test that all models have the same dimensions (for now)."""
        dimensions_set = set()
        for model_key, config in EMBEDDING_MODELS.items():
            dimensions_set.add(config["files"]["dimensions"])
        
        # Currently all models should have 384 dimensions
        assert len(dimensions_set) == 1, "Models have inconsistent dimensions"
        assert list(dimensions_set)[0] == 384


class TestModelConfigsIntegration:
    """Integration tests for model configuration with environment variables."""
    
    def setup_method(self):
        """Set up test environment."""
        # Clear any environment variables
        if 'FUZZYSHELL_MODEL' in os.environ:
            del os.environ['FUZZYSHELL_MODEL']
    
    def test_model_switching_workflow(self):
        """Test complete model switching workflow."""
        # Start with default model
        assert get_active_model_key() == "terminal-minilm-l6"
        config1 = get_model_config()
        assert config1["files"]["tokenizer_type"] == "bert"
        
        # Switch to different model
        with patch.dict(os.environ, {'FUZZYSHELL_MODEL': 'stock-minilm-l6'}):
            assert get_active_model_key() == "stock-minilm-l6"
            config2 = get_model_config()
            assert config2["files"]["tokenizer_type"] == "stock-bert"
            
            # Verify configurations are different
            assert config1["repo"] != config2["repo"]
            assert config1["files"]["tokenizer_type"] != config2["files"]["tokenizer_type"]
    
    def test_invalid_environment_model(self):
        """Test behavior with invalid model in environment."""
        with patch.dict(os.environ, {'FUZZYSHELL_MODEL': 'invalid-model'}):
            with pytest.raises(ValueError):
                get_model_config()


