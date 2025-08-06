"""
Test that embeddings are stored and retrieved correctly - what goes in comes out.
This is a permanent fixture to prevent regressions.
"""
import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
from fuzzyshell import FuzzyShell
from test_helpers import create_test_db_connection

@pytest.fixture
def memory_db():
    """Clean in-memory database for testing"""
    conn = create_test_db_connection()
    yield conn
    conn.close()

@pytest.fixture  
def mock_model():
    """Mock model that returns predictable embeddings"""
    with patch('fuzzyshell.fuzzyshell.ModelHandler') as MockModelHandler:
        mock = MagicMock()
        MockModelHandler.return_value = mock
        
        # Return predictable embeddings based on command content
        def mock_encode(texts):
            embeddings = []
            for text in texts:
                # Create predictable embedding based on hash of text
                hash_val = hash(text) % 1000
                embedding = np.array([hash_val / 1000.0] * 384, dtype=np.float32)
                embeddings.append(embedding)
            return np.array(embeddings)
        
        mock.encode.side_effect = mock_encode
        yield mock

@pytest.fixture
def fs_with_mock_model(memory_db, mock_model):
    """FuzzyShell instance with mocked model for testing"""
    os.environ['FUZZYSHELL_MODEL'] = 'stock-minilm-l6'
    fs = FuzzyShell(conn=memory_db)
    fs.init_model_sync()
    return fs

def test_embedding_roundtrip_basic(fs_with_mock_model, mock_model):
    """Test basic embedding storage and retrieval"""
    # Test data
    test_commands = [
        "ls -lh",
        "git status", 
        "docker ps",
        "python script.py"
    ]
    
    # Store commands with embeddings
    for cmd in test_commands:
        # Get embedding from mock model
        embedding = fs_with_mock_model.model.encode([cmd])[0]
        
        # Store in database
        cmd_id = fs_with_mock_model.command_dal.add_command(cmd, embedding=fs_with_mock_model.quantize_embedding(embedding))
    
    # Retrieve and verify each embedding
    for cmd in test_commands:
        # Get original embedding
        original_embedding = fs_with_mock_model.model.encode([cmd])[0]
        
        # Retrieve from database
        retrieved_commands = fs_with_mock_model.command_dal.get_commands_with_embeddings_matching(f"%{cmd}%")
        
        assert len(retrieved_commands) >= 1, f"Command '{cmd}' not found in database"
        
        # Find exact match
        exact_match = None
        for cmd_id, stored_cmd, stored_emb_blob in retrieved_commands:
            if stored_cmd == cmd:
                exact_match = (cmd_id, stored_cmd, stored_emb_blob)
                break
                
        assert exact_match is not None, f"Exact match for '{cmd}' not found"
        
        cmd_id, stored_cmd, stored_emb_blob = exact_match
        
        # Dequantize stored embedding  
        retrieved_embedding = fs_with_mock_model.dequantize_embedding(stored_emb_blob)
        
        # Verify embeddings match (within quantization tolerance)
        similarity = np.dot(original_embedding, retrieved_embedding) / (
            np.linalg.norm(original_embedding) * np.linalg.norm(retrieved_embedding)
        )
        
        assert similarity > 0.99, f"Embedding mismatch for '{cmd}': similarity={similarity:.3f}"

def test_embedding_quantization_preserves_similarity(fs_with_mock_model):
    """Test that quantization/dequantization preserves relative similarities"""
    # Create embeddings with known relative similarities
    base_embedding = np.random.rand(384).astype(np.float32)
    similar_embedding = base_embedding + 0.1 * np.random.rand(384).astype(np.float32)
    different_embedding = np.random.rand(384).astype(np.float32)
    
    # Normalize
    base_embedding = base_embedding / np.linalg.norm(base_embedding)
    similar_embedding = similar_embedding / np.linalg.norm(similar_embedding)
    different_embedding = different_embedding / np.linalg.norm(different_embedding)
    
    # Calculate original similarities
    orig_sim_similar = np.dot(base_embedding, similar_embedding)
    orig_sim_different = np.dot(base_embedding, different_embedding)
    
    # Quantize and dequantize
    base_quant = fs_with_mock_model.quantize_embedding(base_embedding)
    similar_quant = fs_with_mock_model.quantize_embedding(similar_embedding)
    different_quant = fs_with_mock_model.quantize_embedding(different_embedding)
    
    base_dequant = fs_with_mock_model.dequantize_embedding(base_quant)
    similar_dequant = fs_with_mock_model.dequantize_embedding(similar_quant)
    different_dequant = fs_with_mock_model.dequantize_embedding(different_quant)
    
    # Calculate similarities after quantization
    quant_sim_similar = np.dot(base_dequant, similar_dequant)
    quant_sim_different = np.dot(base_dequant, different_dequant)
    
    # Verify relative order is preserved
    assert orig_sim_similar > orig_sim_different, "Original similarity order incorrect"
    assert quant_sim_similar > quant_sim_different, "Quantized similarity order not preserved"
    
    # Verify quantization error is reasonable
    sim_error = abs(orig_sim_similar - quant_sim_similar)
    diff_error = abs(orig_sim_different - quant_sim_different)
    
    assert sim_error < 0.1, f"Quantization error too large for similar embeddings: {sim_error}"
    assert diff_error < 0.1, f"Quantization error too large for different embeddings: {diff_error}"

def test_search_finds_stored_commands(fs_with_mock_model, mock_model):
    """Test that search can find commands that were stored"""
    # Store some test commands
    test_commands = [
        ("ls -lh", "list files in long format"),
        ("git status", "check git repository status"), 
        ("docker ps", "show running containers"),
        ("python script.py", "run python script")
    ]
    
    for cmd, description in test_commands:
        embedding = fs_with_mock_model.model.encode([cmd])[0]
        cmd_id = fs_with_mock_model.command_dal.add_command(cmd, embedding=fs_with_mock_model.quantize_embedding(embedding))
    
    # Update corpus stats
    fs_with_mock_model.update_corpus_stats()
    
    # Test that each command can be found
    for cmd, description in test_commands:
        results = fs_with_mock_model.search(cmd, top_k=5)
        
        # The exact command should be in the results
        found_exact = any(result[0] == cmd for result in results)
        assert found_exact, f"Exact command '{cmd}' not found in search results: {[r[0] for r in results]}"
        
        # The exact command should be first (with exact match boost)
        assert results[0][0] == cmd, f"Exact command '{cmd}' not first in results. Got: {results[0][0]}"

def test_database_schema_consistency(fs_with_mock_model):
    """Test that database schema supports embedding operations"""
    with fs_with_mock_model.command_dal.connection() as conn:
        c = conn.cursor()
        
        # Check that commands table exists with required columns
        c.execute("PRAGMA table_info(commands)")
        commands_cols = {col[1] for col in c.fetchall()}
        required_cmd_cols = {'id', 'command', 'length'}
        assert required_cmd_cols.issubset(commands_cols), f"Missing required columns in commands table: {required_cmd_cols - commands_cols}"
        
        # Check that embeddings can be stored and retrieved
        # This tests the actual schema that the DAL expects
        test_cmd = "test command"
        test_embedding = np.random.rand(384).astype(np.float32)
        cmd_id = fs_with_mock_model.command_dal.add_command(test_cmd, embedding=fs_with_mock_model.quantize_embedding(test_embedding))
        
        # Verify we can retrieve commands with embeddings
        retrieved = fs_with_mock_model.command_dal.get_commands_with_embeddings_matching("%test%")
        assert len(retrieved) >= 1, "Could not retrieve stored command with embedding"
        
        # Verify embedding data is valid
        cmd_id_ret, cmd_ret, emb_blob = retrieved[0]
        assert cmd_ret == test_cmd, "Retrieved command doesn't match stored command"
        assert emb_blob is not None, "Retrieved embedding blob is None"
        
        # Verify embedding can be dequantized
        retrieved_embedding = fs_with_mock_model.dequantize_embedding(emb_blob)
        assert retrieved_embedding.shape == (384,), f"Dequantized embedding has wrong shape: {retrieved_embedding.shape}"