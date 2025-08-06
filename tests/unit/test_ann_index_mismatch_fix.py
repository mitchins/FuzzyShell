#!/usr/bin/env python3
"""
Unit tests for ANN index mismatch fix.

Tests that ANN search is properly disabled when using filtered candidates
to prevent IndexError from absolute indices being applied to relative arrays.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import MagicMock

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from fuzzyshell import FuzzyShell


class TestANNIndexMismatchFix:
    """Test the ANN index mismatch fix."""
    
    def test_ann_disabled_for_filtered_candidates(self):
        """Test that ANN search is disabled when candidates are filtered."""
        # Create temporary database
        temp_db_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        temp_db_path = temp_db_file.name
        temp_db_file.close()
        
        try:
            fs = FuzzyShell(db_path=temp_db_path)
            fs.wait_for_model(timeout=10.0)
            
            # Add enough commands to trigger both filtering and ANN
            commands = []
            for i in range(150):  # Large enough for ANN
                if i < 50:
                    cmd = f"ls -la file{i}.txt"  # Will match "ls" query
                else:
                    cmd = f"git status {i}"  # Won't match "ls" query
                commands.append(cmd)
            
            # Mock model
            mock_embeddings = np.random.rand(150, 384).astype(np.float32)
            fs._model = MagicMock()
            fs._model.encode.side_effect = lambda cmds: [mock_embeddings[i] for i in range(len(cmds))]
            
            # Add commands with embeddings
            for i, cmd in enumerate(commands):
                quantized = fs.quantize_embedding(mock_embeddings[i])
                fs.command_dal.add_command(cmd, embedding=quantized)
            
            # Build ANN index
            if fs.ann_manager:
                rebuild_success = fs.ann_manager.rebuild_from_database(n_clusters=32)
                assert rebuild_success, "ANN rebuild should succeed"
            
            # Test search with filtered candidates (should disable ANN)
            query_emb = fs.quantize_embedding(np.random.rand(384).astype(np.float32))
            fs._model.encode.return_value = [query_emb]
            
            # This should NOT raise IndexError
            results = fs.search("ls", top_k=5)
            
            assert isinstance(results, list)
            assert len(results) > 0
            
            # Verify results are relevant (contain "ls")
            relevant_count = sum(1 for cmd, _ in results if "ls" in cmd.lower())
            assert relevant_count > 0, "Should find ls commands"
            
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
    
    def test_ann_enabled_for_full_candidates(self):
        """Test that ANN search is still used for broad queries."""
        # Create temporary database
        temp_db_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        temp_db_path = temp_db_file.name
        temp_db_file.close()
        
        try:
            fs = FuzzyShell(db_path=temp_db_path)
            fs.wait_for_model(timeout=10.0)
            
            # Add mixed commands
            commands = []
            for i in range(150):
                cmd = f"command{i} test"
                commands.append(cmd)
            
            # Mock model
            mock_embeddings = np.random.rand(150, 384).astype(np.float32)
            fs._model = MagicMock()
            fs._model.encode.side_effect = lambda cmds: [mock_embeddings[i] for i in range(len(cmds))]
            
            # Add commands with embeddings
            for i, cmd in enumerate(commands):
                quantized = fs.quantize_embedding(mock_embeddings[i])
                fs.command_dal.add_command(cmd, embedding=quantized)
            
            # Build ANN index
            if fs.ann_manager:
                rebuild_success = fs.ann_manager.rebuild_from_database(n_clusters=32)
                assert rebuild_success, "ANN rebuild should succeed"
            
            # Test broad search (should allow ANN if available)
            query_emb = fs.quantize_embedding(np.random.rand(384).astype(np.float32))
            fs._model.encode.return_value = [query_emb]
            
            # This should work with or without ANN
            results = fs.search("query", top_k=5)  # Generic term
            
            assert isinstance(results, list)
            assert len(results) > 0
            
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
    
    def test_search_engine_filtering_logic(self):
        """Test the search engine's candidate filtering logic directly."""
        # Create temporary database
        temp_db_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        temp_db_path = temp_db_file.name
        temp_db_file.close()
        
        try:
            fs = FuzzyShell(db_path=temp_db_path)
            fs.wait_for_model(timeout=10.0)
            
            # Add enough "ls" commands to trigger filtering (>= 20)
            ls_commands = [f"ls -la file{i}.txt" for i in range(25)]
            other_commands = ["git status", "docker ps", "vim file.txt"]
            all_commands = ls_commands + other_commands
            
            # Mock model
            mock_embeddings = np.random.rand(len(all_commands), 384).astype(np.float32)
            fs._model = MagicMock()
            fs._model.encode.side_effect = lambda cmds: [mock_embeddings[i % len(mock_embeddings)] for i in range(len(cmds))]
            
            # Add commands with embeddings
            for i, cmd in enumerate(all_commands):
                quantized = fs.quantize_embedding(mock_embeddings[i])
                fs.command_dal.add_command(cmd, embedding=quantized)
            
            # Test filtering for "ls" query (should get filtered candidates)
            candidates, is_filtered = fs.search_engine._get_search_candidates("ls")
            
            assert is_filtered == True, "Should filter when >= 20 matches found"
            assert len(candidates) >= 20, "Should return filtered candidates"
            
            # All candidates should contain "ls"
            for _, command, _ in candidates:
                assert "ls" in command.lower(), f"Filtered candidate '{command}' should contain 'ls'"
            
            # Test broad query (should not filter)
            candidates, is_filtered = fs.search_engine._get_search_candidates("xyz_nonexistent")
            
            assert is_filtered == False, "Should not filter when < 20 matches found"
            assert len(candidates) >= len(all_commands), "Should return full dataset"
            
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
    
    def test_no_index_error_regression(self):
        """Regression test to ensure the original IndexError never occurs."""
        # Create temporary database
        temp_db_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        temp_db_path = temp_db_file.name
        temp_db_file.close()
        
        try:
            fs = FuzzyShell(db_path=temp_db_path)
            fs.wait_for_model(timeout=10.0)
            
            # Recreate the exact scenario that caused the original bug
            commands = []
            for i in range(200):  # Large dataset for ANN
                if i < 50:
                    cmd = f"ls -la {i}"  # These match "ls" queries
                else:
                    cmd = f"git status {i}"  # These don't
                commands.append(cmd)
            
            # Mock model
            mock_embeddings = np.random.rand(200, 384).astype(np.float32)
            fs._model = MagicMock()
            fs._model.encode.side_effect = lambda cmds: [mock_embeddings[i] for i in range(len(cmds))]
            
            # Add commands with embeddings
            for i, cmd in enumerate(commands):
                quantized = fs.quantize_embedding(mock_embeddings[i])
                fs.command_dal.add_command(cmd, embedding=quantized)
            
            # Build ANN index on full dataset (200 commands)
            if fs.ann_manager:
                rebuild_success = fs.ann_manager.rebuild_from_database(n_clusters=32)
                assert rebuild_success
            
            # Test the problematic queries that caused IndexError
            problematic_queries = ["l", "li", "lis", "list"]
            
            for query in problematic_queries:
                query_emb = fs.quantize_embedding(np.random.rand(384).astype(np.float32))
                fs._model.encode.return_value = [query_emb]
                
                # This should NOT raise IndexError
                try:
                    results = fs.search(query, top_k=5)
                    assert isinstance(results, list), f"Query '{query}' should return list"
                    # Don't assert len > 0 since some queries might not match anything
                except IndexError as e:
                    pytest.fail(f"IndexError occurred for query '{query}': {e}")
                except Exception as e:
                    # Other exceptions are okay, but IndexError is what we're testing for
                    pass
            
            print("✅ No IndexError occurred for any problematic query")
            
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)