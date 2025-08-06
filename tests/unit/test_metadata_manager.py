#!/usr/bin/env python3
"""
Unit tests for MetadataManager class.

Tests metadata operations including database information gathering,
scoring preferences, model configuration, and system diagnostics.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from test_helpers import create_test_db_connection

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from fuzzyshell.metadata_manager import MetadataManager
from fuzzyshell.data.datastore import MetadataDAL, MockDatabaseProvider


@pytest.fixture
def metadata_dal():
    """Create MetadataDAL with test database."""
    conn = create_test_db_connection()
    db_provider = MockDatabaseProvider(conn)
    return MetadataDAL(db_provider)


@pytest.fixture
def mock_fuzzyshell():
    """Create mock FuzzyShell instance."""
    mock_fs = Mock()
    mock_fs.get_indexed_count.return_value = 1234
    mock_fs.db_path = "/test/path/fuzzyshell.db"
    mock_fs._model = Mock()
    mock_fs.search_engine = Mock()
    
    # Mock command_dal for diagnostics
    mock_fs.command_dal = Mock()
    mock_fs.command_dal.get_embedding_count.return_value = 1000
    mock_fs.command_dal.get_command_count.return_value = 1234
    
    return mock_fs


@pytest.fixture
def metadata_manager(metadata_dal, mock_fuzzyshell):
    """Create MetadataManager with test dependencies."""
    return MetadataManager(metadata_dal, mock_fuzzyshell)


@pytest.fixture
def metadata_manager_no_fs(metadata_dal):
    """Create MetadataManager without FuzzyShell instance."""
    return MetadataManager(metadata_dal, None)


def test_metadata_manager_initialization(metadata_dal, mock_fuzzyshell):
    """Test MetadataManager initializes correctly."""
    manager = MetadataManager(metadata_dal, mock_fuzzyshell)
    assert manager.metadata_dal == metadata_dal
    assert manager.fuzzyshell == mock_fuzzyshell


def test_get_database_info_with_fuzzyshell(metadata_manager, metadata_dal):
    """Test getting database info with FuzzyShell instance."""
    # Set up test metadata
    metadata_dal.set('model_name', 'test-model')
    metadata_dal.set('last_ann_build', '2023-12-01')
    
    db_info = metadata_manager.get_database_info()
    
    assert isinstance(db_info, dict)
    assert db_info['item_count'] == 1234
    assert db_info['embedding_model'] == 'test-model'
    assert db_info['last_updated'] == '2023-12-01'
    assert db_info['db_path'] == "/test/path/fuzzyshell.db"
    assert 'uuid' in db_info
    assert 'created' in db_info


def test_get_database_info_without_fuzzyshell(metadata_manager_no_fs, metadata_dal):
    """Test getting database info without FuzzyShell instance."""
    db_info = metadata_manager_no_fs.get_database_info()
    
    assert isinstance(db_info, dict)
    assert 'uuid' in db_info
    assert 'created' in db_info
    # Should not have FuzzyShell-specific info
    assert 'item_count' not in db_info


def test_get_database_info_handles_errors(metadata_manager):
    """Test database info gracefully handles errors."""
    # Make FuzzyShell methods raise exceptions
    metadata_manager.fuzzyshell.get_indexed_count.side_effect = Exception("Test error")
    
    db_info = metadata_manager.get_database_info()
    
    # Should still return basic database info despite errors
    assert isinstance(db_info, dict)
    assert 'uuid' in db_info


def test_get_scoring_configuration(metadata_manager, metadata_dal):
    """Test getting scoring configuration."""
    metadata_dal.set_scoring_preference('more_semantic')
    
    config = metadata_manager.get_scoring_configuration()
    
    assert config['preference'] == 'more_semantic'
    assert config['semantic_weight'] == 0.7
    assert config['bm25_weight'] == 0.3
    assert 'available_preferences' in config
    assert len(config['available_preferences']) == 4


def test_get_scoring_configuration_default(metadata_manager):
    """Test scoring configuration returns defaults."""
    config = metadata_manager.get_scoring_configuration()
    
    assert config['preference'] == 'balanced'
    assert config['semantic_weight'] == 0.5
    assert config['bm25_weight'] == 0.5


@pytest.mark.parametrize("preference,expected_semantic,expected_bm25", [
    ('less_semantic', 0.3, 0.7),
    ('balanced', 0.5, 0.5),
    ('more_semantic', 0.7, 0.3),
    ('semantic_only', 1.0, 0.0)
])
def test_set_scoring_preference_valid(metadata_manager, preference, expected_semantic, expected_bm25):
    """Test setting valid scoring preferences."""
    result = metadata_manager.set_scoring_preference(preference)
    
    assert result is True
    
    # Verify it was actually set
    config = metadata_manager.get_scoring_configuration()
    assert config['preference'] == preference
    assert abs(config['semantic_weight'] - expected_semantic) < 0.1
    assert abs(config['bm25_weight'] - expected_bm25) < 0.1


def test_set_scoring_preference_invalid(metadata_manager):
    """Test setting invalid scoring preference."""
    result = metadata_manager.set_scoring_preference('invalid_preference')
    
    assert result is False
    
    # Should still be default
    config = metadata_manager.get_scoring_configuration()
    assert config['preference'] == 'balanced'


def test_get_model_configuration(metadata_manager, metadata_dal):
    """Test getting model configuration."""
    # Set up test model metadata
    metadata_dal.set('model_name', 'test-model-v1')
    metadata_dal.set('model_repo', 'huggingface/test-model')
    metadata_dal.set('model_dimensions', '384')
    metadata_dal.set('model_tokenizer_type', 'sentencepiece')
    metadata_dal.set('model_key', 'test-key')
    
    config = metadata_manager.get_model_configuration()
    
    assert config['model_name'] == 'test-model-v1'
    assert config['model_repo'] == 'huggingface/test-model'
    assert config['model_dimensions'] == '384'
    assert config['model_tokenizer_type'] == 'sentencepiece'
    assert config['model_key'] == 'test-key'


def test_get_model_configuration_defaults(metadata_manager):
    """Test model configuration returns defaults for missing values."""
    config = metadata_manager.get_model_configuration()
    
    assert config['model_name'] == 'unknown'
    assert config['model_repo'] == 'unknown'
    assert config['model_dimensions'] == 'unknown'
    assert config['model_tokenizer_type'] == 'unknown'
    assert config['model_key'] == 'unknown'


def test_update_model_metadata(metadata_manager, metadata_dal):
    """Test updating model metadata."""
    test_config = {
        'repo': 'huggingface/new-model',
        'files': {
            'dimensions': 512,
            'tokenizer_type': 'wordpiece'
        }
    }
    
    result = metadata_manager.update_model_metadata('new-model-key', test_config)
    
    assert result is True
    
    # Verify metadata was updated
    assert metadata_dal.get('model_name') == 'new-model-key'
    assert metadata_dal.get('model_repo') == 'huggingface/new-model'
    assert metadata_dal.get('model_dimensions') == '512'
    assert metadata_dal.get('model_tokenizer_type') == 'wordpiece'
    assert metadata_dal.get('model_key') == 'new-model-key'


def test_update_model_metadata_partial_config(metadata_manager, metadata_dal):
    """Test updating model metadata with incomplete config."""
    test_config = {
        'repo': 'test/partial-config'
        # Missing 'files' section
    }
    
    result = metadata_manager.update_model_metadata('partial-model', test_config)
    
    assert result is True
    
    # Should handle missing values gracefully
    assert metadata_dal.get('model_name') == 'partial-model'
    assert metadata_dal.get('model_repo') == 'test/partial-config'
    assert metadata_dal.get('model_dimensions') == 'unknown'
    assert metadata_dal.get('model_tokenizer_type') == 'unknown'


def test_get_system_diagnostics(metadata_manager):
    """Test system diagnostics gathering."""
    diagnostics = metadata_manager.get_system_diagnostics()
    
    assert isinstance(diagnostics, dict)
    assert diagnostics['database_healthy'] is True
    assert diagnostics['model_loaded'] is True
    assert diagnostics['search_engine_ready'] is True
    assert diagnostics['item_count'] == 1234
    assert diagnostics['embedding_count'] == 1000
    assert diagnostics['total_commands'] == 1234


def test_get_system_diagnostics_no_fuzzyshell(metadata_manager_no_fs):
    """Test system diagnostics without FuzzyShell instance."""
    diagnostics = metadata_manager_no_fs.get_system_diagnostics()
    
    assert isinstance(diagnostics, dict)
    assert diagnostics['database_healthy'] is True
    assert diagnostics['model_loaded'] is False
    assert diagnostics['search_engine_ready'] is False


def test_get_system_diagnostics_handles_errors(metadata_manager):
    """Test diagnostics handles errors gracefully."""
    # Make command_dal methods raise errors
    metadata_manager.fuzzyshell.command_dal.get_embedding_count.side_effect = Exception("Test error")
    
    diagnostics = metadata_manager.get_system_diagnostics()
    
    # Should still return basic diagnostics
    assert isinstance(diagnostics, dict)
    assert diagnostics['database_healthy'] is True
    # Should not have embedding statistics due to error
    assert 'embedding_count' not in diagnostics


def test_get_expert_screen_data(metadata_manager, metadata_dal):
    """Test getting comprehensive expert screen data."""
    # Set up some test metadata
    metadata_dal.set('model_name', 'expert-test-model')
    metadata_dal.set_scoring_preference('more_semantic')
    
    data = metadata_manager.get_expert_screen_data()
    
    assert isinstance(data, dict)
    assert 'database' in data
    assert 'scoring' in data
    assert 'model' in data
    assert 'diagnostics' in data
    assert 'ann_config' in data
    assert 'version' in data
    
    # Check nested data structure
    assert data['database']['item_count'] == 1234
    assert data['scoring']['preference'] == 'more_semantic'
    assert data['model']['model_name'] == 'expert-test-model'
    assert data['diagnostics']['model_loaded'] is True
    assert data['ann_config']['num_clusters'] == 32


def test_get_expert_screen_data_with_imports(metadata_manager):
    """Test expert screen data handles imports gracefully."""
    # This test ensures the method doesn't crash if imports are available
    data = metadata_manager.get_expert_screen_data()
    
    assert 'ann_config' in data
    assert isinstance(data['ann_config'], dict)
    # The actual imported values depend on the module state, but should have defaults
    assert 'num_clusters' in data['ann_config']
    assert 'use_ann_search' in data['ann_config']


def test_set_metadata(metadata_manager, metadata_dal):
    """Test setting arbitrary metadata."""
    result = metadata_manager.set_metadata('test_key', 'test_value')
    
    assert result is True
    assert metadata_dal.get('test_key') == 'test_value'


def test_set_metadata_handles_errors(metadata_manager):
    """Test set_metadata handles errors gracefully."""
    # Mock the metadata_dal to raise an error
    metadata_manager.metadata_dal.set = Mock(side_effect=Exception("Test error"))
    
    result = metadata_manager.set_metadata('failing_key', 'failing_value')
    
    assert result is False


def test_get_metadata(metadata_manager, metadata_dal):
    """Test getting arbitrary metadata."""
    metadata_dal.set('existing_key', 'existing_value')
    
    # Test existing key
    value = metadata_manager.get_metadata('existing_key')
    assert value == 'existing_value'
    
    # Test non-existing key with default
    value = metadata_manager.get_metadata('missing_key', 'default_value')
    assert value == 'default_value'
    
    # Test non-existing key without default
    value = metadata_manager.get_metadata('missing_key')
    assert value is None


def test_get_metadata_handles_errors(metadata_manager):
    """Test get_metadata handles errors gracefully."""
    # Mock the metadata_dal to raise an error
    metadata_manager.metadata_dal.get = Mock(side_effect=Exception("Test error"))
    
    value = metadata_manager.get_metadata('failing_key', 'fallback')
    assert value == 'fallback'


def test_version_fallback(metadata_manager):
    """Test version fallback when import fails."""
    with patch('fuzzyshell.metadata_manager.logger'):
        # Simulate import failure by patching the import
        with patch.dict('sys.modules', {'fuzzyshell.__version__': None}):
            version = metadata_manager._get_version()
            assert version == "0.1.0"


def test_expert_screen_data_handles_errors(metadata_manager):
    """Test expert screen data handles errors gracefully."""
    # Make database info raise an error
    metadata_manager.get_database_info = Mock(side_effect=Exception("Database error"))
    
    data = metadata_manager.get_expert_screen_data()
    
    assert isinstance(data, dict)
    assert 'error' in data