"""
Tests for model change detection and metadata handling.
"""

import os
import tempfile
import sqlite3
from unittest.mock import patch, MagicMock


from test_helpers import create_test_db_connection
from fuzzyshell.fuzzyshell import FuzzyShell


class TestModelChangeDetection:
    """Test model change detection functionality."""

    def setup_method(self):
        """Set up test environment with in-memory database."""
        # Clear any environment variables
        if 'FUZZYSHELL_MODEL' in os.environ:
            del os.environ['FUZZYSHELL_MODEL']
        
        # Create test database connection
        self.test_conn = create_test_db_connection()
        
        # Create FuzzyShell instance with test connection
        self.fuzzyshell = FuzzyShell(conn=self.test_conn)
    
    def test_initial_model_metadata_storage(self):
        """Test that model metadata is stored on first run."""
        # Initially no model metadata should exist
        assert self.fuzzyshell.get_metadata('model_name') is None
        
        # Store model metadata
        self.fuzzyshell.store_model_metadata('terminal-minilm-l6')
        
        # Verify metadata was stored
        assert self.fuzzyshell.get_metadata('model_name') == 'terminal-minilm-l6'
        assert (
            self.fuzzyshell.get_metadata('model_repo')
            == 'Mitchins/minilm-l6-v2-terminal-describer-embeddings-ONNX'
        )
        assert self.fuzzyshell.get_metadata('model_dimensions') == '384'
        assert self.fuzzyshell.get_metadata('model_tokenizer_type') == 'bert'
    
    def test_store_model_metadata_with_none(self):
        """Test storing model metadata with None uses active model."""
        with patch.dict(os.environ, {'FUZZYSHELL_MODEL': 'terminal-minilm-l6'}):
            self.fuzzyshell.store_model_metadata(None)
            
            assert self.fuzzyshell.get_metadata('model_name') == 'terminal-minilm-l6'
            assert self.fuzzyshell.get_metadata('model_tokenizer_type') == 'bert'
    
    def test_detect_model_change_no_change(self):
        """Test model change detection when no change occurred."""
        # Set initial model metadata
        self.fuzzyshell.store_model_metadata('terminal-minilm-l6')
        
        # Check for changes (should be none with default model)
        changed, current, stored = self.fuzzyshell.detect_model_change()
        
        assert not changed
        assert current == 'terminal-minilm-l6'
        assert stored == 'terminal-minilm-l6'
    
    def test_detect_model_change_with_change(self):
        """Test model change detection when change occurred."""
        # Set initial model metadata
        self.fuzzyshell.store_model_metadata('terminal-minilm-l6')
        
        # Change environment to different model
        with patch.dict(os.environ, {'FUZZYSHELL_MODEL': 'stock-minilm-l6'}):
            changed, current, stored = self.fuzzyshell.detect_model_change()
            
            assert changed
            assert current == 'stock-minilm-l6'
            assert stored == 'terminal-minilm-l6'
    
    def test_detect_model_change_first_run(self):
        """Test model change detection on first run (no stored metadata)."""
        # No metadata stored initially
        changed, current, stored = self.fuzzyshell.detect_model_change()
        
        # Should not detect change when both are the same default
        assert not changed
        assert current == 'terminal-minilm-l6'  # Default active
        assert stored == 'terminal-minilm-l6'   # Default fallback
    
    def test_handle_model_change_no_change(self):
        """Test handle_model_change when no change occurred."""
        # Set initial model metadata  
        self.fuzzyshell.store_model_metadata('terminal-minilm-l6')
        
        # Add some test data
        self.fuzzyshell.add_command("test command")
        
        # Handle model change (should be none)
        result = self.fuzzyshell.handle_model_change()
        
        assert not result
        
        # Verify data is still there
        c = self.test_conn.cursor()
        c.execute("SELECT COUNT(*) FROM commands")
        assert c.fetchone()[0] == 1
    
    def test_handle_model_change_with_change(self):
        """Test handle_model_change when change occurred."""
        # Set initial model metadata
        self.fuzzyshell.store_model_metadata('terminal-minilm-l6')
        
        # Add some test data and embeddings
        self.fuzzyshell.add_command("test command")
        c = self.test_conn.cursor()
        c.execute("INSERT INTO embeddings (embedding) VALUES (?)", 
                 (b'fake_embedding_data',))
        self.test_conn.commit()
        
        # Verify data exists
        c.execute("SELECT COUNT(*) FROM commands")
        assert c.fetchone()[0] == 1
        c.execute("SELECT COUNT(*) FROM embeddings")
        assert c.fetchone()[0] == 1
        
        # Change environment to different model
        with patch.dict(os.environ, {'FUZZYSHELL_MODEL': 'stock-minilm-l6'}):
            result = self.fuzzyshell.handle_model_change()
            
            assert result

            # Verify metadata was updated
            assert self.fuzzyshell.get_metadata('model_name') == 'stock-minilm-l6'
            assert self.fuzzyshell.get_metadata('model_tokenizer_type') == 'stock-bert'

            # Verify embeddings were cleared but commands remain
            c.execute("SELECT COUNT(*) FROM commands")
            assert c.fetchone()[0] == 1  # Commands should remain
            c.execute("SELECT COUNT(*) FROM embeddings")
            assert c.fetchone()[0] == 0  # Embeddings should be cleared

            # Verify ANN counters were reset
            assert self.fuzzyshell.get_metadata('ann_command_count') == '0'
            assert self.fuzzyshell.get_metadata('poorly_clustered_commands') == '0'
    
    def test_clear_embeddings(self):
        """Test clearing embeddings from database."""
        # Add test embeddings
        c = self.test_conn.cursor()
        c.execute("INSERT INTO commands (command) VALUES (?)", ("test1",))
        c.execute("INSERT INTO commands (command) VALUES (?)", ("test2",))
        c.execute("INSERT INTO embeddings (embedding) VALUES (?)", 
                 (b'embedding1',))
        c.execute("INSERT INTO embeddings (embedding) VALUES (?)", 
                 (b'embedding2',))
        self.test_conn.commit()
        
        # Verify embeddings exist
        c.execute("SELECT COUNT(*) FROM embeddings")
        assert c.fetchone()[0] == 2
        
        # Clear embeddings
        self.fuzzyshell.clear_embeddings()
        
        # Verify embeddings were cleared but commands remain
        c.execute("SELECT COUNT(*) FROM embeddings")
        assert c.fetchone()[0] == 0
        c.execute("SELECT COUNT(*) FROM commands")
        assert c.fetchone()[0] == 2
    
    def test_model_metadata_storage_different_models(self):
        """Test storing metadata for different models."""
        # Store metadata for multilingual model
        self.fuzzyshell.store_model_metadata('terminal-minilm-l6')
        
        assert self.fuzzyshell.get_metadata('model_name') == 'terminal-minilm-l6'
        assert self.fuzzyshell.get_metadata('model_tokenizer_type') == 'bert'
        
        # Store metadata for minilm model
        self.fuzzyshell.store_model_metadata('terminal-minilm-l6')
        
        assert self.fuzzyshell.get_metadata('model_name') == 'terminal-minilm-l6'
        assert self.fuzzyshell.get_metadata('model_tokenizer_type') == 'bert'
        
        # Verify repo changed
        repo = self.fuzzyshell.get_metadata('model_repo')
        assert 'minilm-l6' in repo
    
    def test_integration_with_environment_changes(self):
        """Test complete integration with environment variable changes."""
        # Start with default model
        self.fuzzyshell.store_model_metadata()
        initial_model = self.fuzzyshell.get_metadata('model_name')
        
        # Add some data
        self.fuzzyshell.add_command("test command")
        c = self.test_conn.cursor()
        c.execute("INSERT INTO embeddings (embedding) VALUES (?)", 
                 (b'embedding_data',))
        self.test_conn.commit()
        
        # Change environment and handle change
        with patch.dict(os.environ, {'FUZZYSHELL_MODEL': 'stock-minilm-l6'}):
            changed = self.fuzzyshell.handle_model_change()
            
            assert changed
            assert self.fuzzyshell.get_metadata('model_name') == 'stock-minilm-l6'
            assert self.fuzzyshell.get_metadata('model_name') != initial_model

            # Verify cleanup occurred
            c.execute("SELECT COUNT(*) FROM embeddings")
            assert c.fetchone()[0] == 0


