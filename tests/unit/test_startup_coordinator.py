#!/usr/bin/env python3
"""
Unit tests for StartupCoordinator class.

Tests model change detection, metadata management, and startup coordination
including cache clearing, embedding cleanup, and error handling.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from fuzzyshell.startup_coordinator import StartupCoordinator


@pytest.fixture
def mock_fuzzyshell():
    """Create mock FuzzyShell instance with all required attributes."""
    mock_fs = Mock()
    
    # Mock DAL objects
    mock_fs.metadata_dal = Mock()
    mock_fs.command_dal = Mock()
    mock_fs.cache_dal = Mock()
    mock_fs.ann_manager = Mock()
    
    # Mock metadata methods
    mock_fs.get_metadata = Mock()
    mock_fs.set_metadata = Mock()
    mock_fs.clear_embeddings = Mock()
    mock_fs.clear_all_caches = Mock()
    
    return mock_fs


@pytest.fixture
def startup_coordinator(mock_fuzzyshell):
    """Create StartupCoordinator with mock FuzzyShell."""
    return StartupCoordinator(mock_fuzzyshell)


def test_initialization(mock_fuzzyshell):
    """Test StartupCoordinator initializes correctly."""
    coordinator = StartupCoordinator(mock_fuzzyshell)
    assert coordinator.fuzzyshell == mock_fuzzyshell


def test_store_model_metadata_default_key(startup_coordinator):
    """Test storing model metadata with default key."""
    with patch('fuzzyshell.startup_coordinator.get_active_model_key') as mock_get_key:
        with patch('fuzzyshell.startup_coordinator.get_model_config') as mock_get_config:
            mock_get_key.return_value = 'test-model'
            mock_get_config.return_value = {
                'repo': 'test/model-repo',
                'files': {
                    'dimensions': 512,
                    'tokenizer_type': 'bert'
                }
            }
            
            startup_coordinator.store_model_metadata()
    
    # Verify metadata was set correctly
    startup_coordinator.fuzzyshell.metadata_dal.set.assert_any_call('model_name', 'test-model')
    startup_coordinator.fuzzyshell.metadata_dal.set.assert_any_call('model_repo', 'test/model-repo')
    startup_coordinator.fuzzyshell.metadata_dal.set.assert_any_call('model_dimensions', '512')
    startup_coordinator.fuzzyshell.metadata_dal.set.assert_any_call('model_tokenizer_type', 'bert')


def test_store_model_metadata_custom_key(startup_coordinator):
    """Test storing model metadata with custom key."""
    with patch('fuzzyshell.startup_coordinator.get_model_config') as mock_get_config:
        mock_get_config.return_value = {
            'repo': 'custom/model',
            'files': {
                'dimensions': 768,
                'tokenizer_type': 'roberta'
            }
        }
        
        startup_coordinator.store_model_metadata('custom-model')
    
    # Verify custom model was used
    startup_coordinator.fuzzyshell.metadata_dal.set.assert_any_call('model_name', 'custom-model')
    startup_coordinator.fuzzyshell.metadata_dal.set.assert_any_call('model_repo', 'custom/model')
    startup_coordinator.fuzzyshell.metadata_dal.set.assert_any_call('model_dimensions', '768')
    startup_coordinator.fuzzyshell.metadata_dal.set.assert_any_call('model_tokenizer_type', 'roberta')


def test_handle_model_change_during_init_first_run(startup_coordinator):
    """Test handling model change during init on first run."""
    # Mock first run scenario (no stored model)
    startup_coordinator.fuzzyshell.metadata_dal.get.return_value = None
    
    with patch('fuzzyshell.startup_coordinator.get_active_model_key') as mock_get_key:
        mock_get_key.return_value = 'initial-model'
        
        with patch.object(startup_coordinator, 'store_model_metadata') as mock_store:
            startup_coordinator.handle_model_change_during_init()
    
    # Should store metadata but not clear anything
    mock_store.assert_called_once_with('initial-model')
    startup_coordinator.fuzzyshell.command_dal.clear_embeddings.assert_not_called()
    startup_coordinator.fuzzyshell.cache_dal.clear_cache.assert_not_called()


def test_handle_model_change_during_init_no_change(startup_coordinator):
    """Test handling model change during init when model hasn't changed."""
    # Mock same model scenario
    startup_coordinator.fuzzyshell.metadata_dal.get.return_value = 'current-model'
    
    with patch('fuzzyshell.startup_coordinator.get_active_model_key') as mock_get_key:
        mock_get_key.return_value = 'current-model'
        
        startup_coordinator.handle_model_change_during_init()
    
    # Should not clear anything or update metadata
    startup_coordinator.fuzzyshell.command_dal.clear_embeddings.assert_not_called()
    startup_coordinator.fuzzyshell.cache_dal.clear_cache.assert_not_called()


def test_handle_model_change_during_init_with_change(startup_coordinator):
    """Test handling model change during init when model has changed."""
    # Mock model change scenario
    startup_coordinator.fuzzyshell.metadata_dal.get.return_value = 'old-model'
    
    with patch('fuzzyshell.startup_coordinator.get_active_model_key') as mock_get_key:
        with patch('fuzzyshell.startup_coordinator.tui_safe_print') as mock_print:
            mock_get_key.return_value = 'new-model'
            
            with patch.object(startup_coordinator, 'store_model_metadata') as mock_store:
                startup_coordinator.handle_model_change_during_init()
    
    # Should clear embeddings and caches
    startup_coordinator.fuzzyshell.command_dal.clear_embeddings.assert_called_once()
    startup_coordinator.fuzzyshell.cache_dal.clear_cache.assert_called_once()
    startup_coordinator.fuzzyshell.ann_manager.clear_cache.assert_called_once()
    
    # Should update metadata
    mock_store.assert_called_once_with('new-model')
    
    # Should print user-friendly messages
    assert mock_print.call_count >= 2


def test_handle_model_change_during_init_no_ann_manager(startup_coordinator):
    """Test handling model change during init when ANN manager is None."""
    startup_coordinator.fuzzyshell.metadata_dal.get.return_value = 'old-model'
    startup_coordinator.fuzzyshell.ann_manager = None
    
    with patch('fuzzyshell.startup_coordinator.get_active_model_key') as mock_get_key:
        mock_get_key.return_value = 'new-model'
        
        with patch.object(startup_coordinator, 'store_model_metadata'):
            startup_coordinator.handle_model_change_during_init()
    
    # Should still clear other caches without error
    startup_coordinator.fuzzyshell.command_dal.clear_embeddings.assert_called_once()
    startup_coordinator.fuzzyshell.cache_dal.clear_cache.assert_called_once()


def test_detect_model_change_no_change(startup_coordinator):
    """Test detecting model change when no change occurred."""
    startup_coordinator.fuzzyshell.get_metadata.return_value = 'current-model'
    
    with patch('fuzzyshell.startup_coordinator.get_active_model_key') as mock_get_key:
        mock_get_key.return_value = 'current-model'
        
        changed, current, stored = startup_coordinator.detect_model_change()
    
    assert changed is False
    assert current == 'current-model'
    assert stored == 'current-model'


def test_detect_model_change_with_change(startup_coordinator):
    """Test detecting model change when change occurred."""
    startup_coordinator.fuzzyshell.get_metadata.return_value = 'old-model'
    
    with patch('fuzzyshell.startup_coordinator.get_active_model_key') as mock_get_key:
        mock_get_key.return_value = 'new-model'
        
        changed, current, stored = startup_coordinator.detect_model_change()
    
    assert changed is True
    assert current == 'new-model'
    assert stored == 'old-model'


def test_detect_model_change_default_model(startup_coordinator):
    """Test detecting model change with default model fallback."""
    startup_coordinator.fuzzyshell.get_metadata.return_value = 'terminal-minilm-l6'  # Default fallback
    
    with patch('fuzzyshell.startup_coordinator.get_active_model_key') as mock_get_key:
        mock_get_key.return_value = 'new-model'
        
        changed, current, stored = startup_coordinator.detect_model_change()
    
    assert changed is True
    assert current == 'new-model'
    assert stored == 'terminal-minilm-l6'
    
    # Verify default was used
    startup_coordinator.fuzzyshell.get_metadata.assert_called_with('model_name', 'terminal-minilm-l6')


def test_handle_model_change_no_change(startup_coordinator):
    """Test handling model change when no change occurred."""
    with patch.object(startup_coordinator, 'detect_model_change') as mock_detect:
        mock_detect.return_value = (False, 'current-model', 'current-model')
        
        result = startup_coordinator.handle_model_change()
    
    assert result is False
    startup_coordinator.fuzzyshell.clear_embeddings.assert_not_called()
    startup_coordinator.fuzzyshell.clear_all_caches.assert_not_called()


def test_handle_model_change_with_change(startup_coordinator):
    """Test handling model change when change occurred."""
    with patch.object(startup_coordinator, 'detect_model_change') as mock_detect:
        with patch.object(startup_coordinator, 'store_model_metadata') as mock_store:
            with patch('fuzzyshell.startup_coordinator.tui_safe_print') as mock_print:
                mock_detect.return_value = (True, 'new-model', 'old-model')
                
                result = startup_coordinator.handle_model_change()
    
    assert result is True
    
    # Should clear embeddings and caches
    startup_coordinator.fuzzyshell.clear_embeddings.assert_called_once()
    startup_coordinator.fuzzyshell.clear_all_caches.assert_called_once()
    
    # Should update metadata
    mock_store.assert_called_once_with('new-model')
    
    # Should reset ANN metadata
    startup_coordinator.fuzzyshell.set_metadata.assert_any_call('ann_command_count', '0')
    startup_coordinator.fuzzyshell.set_metadata.assert_any_call('poorly_clustered_commands', '0')
    
    # Should print user messages
    assert mock_print.call_count >= 2


@pytest.mark.parametrize("current_model,stored_model,expected_change", [
    ('model-a', 'model-a', False),
    ('model-a', 'model-b', True),
    ('new-model', None, True),  # First run case
    ('terminal-minilm-l6', 'terminal-minilm-l6', False),
])
def test_detect_model_change_scenarios(startup_coordinator, current_model, stored_model, expected_change):
    """Test various model change detection scenarios."""
    startup_coordinator.fuzzyshell.get_metadata.return_value = stored_model or 'terminal-minilm-l6'
    
    with patch('fuzzyshell.startup_coordinator.get_active_model_key') as mock_get_key:
        mock_get_key.return_value = current_model
        
        changed, current, stored = startup_coordinator.detect_model_change()
    
    assert changed == expected_change
    assert current == current_model


def test_import_fallbacks():
    """Test import fallback functions work correctly."""
    # Test with clean import
    from fuzzyshell.startup_coordinator import get_active_model_key, get_model_config
    
    # Should not raise errors
    model_key = get_active_model_key()
    assert isinstance(model_key, str)
    
    config = get_model_config('terminal-minilm-l6')
    assert isinstance(config, dict)
    assert 'repo' in config


def test_error_handling_in_store_metadata(startup_coordinator):
    """Test error handling in store_model_metadata."""
    with patch('fuzzyshell.startup_coordinator.get_active_model_key') as mock_get_key:
        with patch('fuzzyshell.startup_coordinator.get_model_config') as mock_get_config:
            mock_get_key.return_value = 'test-model'
            mock_get_config.side_effect = Exception("Config error")
            
            # Should not crash even if config fails
            with pytest.raises(Exception):
                startup_coordinator.store_model_metadata()


def test_comprehensive_startup_flow(startup_coordinator):
    """Test comprehensive startup flow with model change."""
    # Simulate full startup with model change
    startup_coordinator.fuzzyshell.metadata_dal.get.return_value = 'old-model'
    
    with patch('fuzzyshell.startup_coordinator.get_active_model_key') as mock_get_key:
        with patch('fuzzyshell.startup_coordinator.get_model_config') as mock_get_config:
            with patch('fuzzyshell.startup_coordinator.tui_safe_print'):
                mock_get_key.return_value = 'new-model'
                mock_get_config.return_value = {
                    'repo': 'test/new-model',
                    'files': {'dimensions': 384, 'tokenizer_type': 'bert'}
                }
                
                # Run initialization phase
                startup_coordinator.handle_model_change_during_init()
                
                # Run runtime phase
                result = startup_coordinator.handle_model_change()
    
    # Both phases should detect and handle the change
    assert startup_coordinator.fuzzyshell.command_dal.clear_embeddings.call_count >= 1
    assert startup_coordinator.fuzzyshell.metadata_dal.set.call_count >= 4  # 4 metadata fields
    assert result is True