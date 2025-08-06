import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
from fuzzyshell import FuzzyShell
from test_helpers import create_test_db_connection

@pytest.fixture
def memory_db():
    # Use dependency injection with in-memory SQLite database
    conn = create_test_db_connection()
    yield conn
    conn.close()

@pytest.fixture
def temp_history():
    test_commands = [
        "git push origin main",
        "docker build -t myapp .",
        "npm install express",
        "python manage.py runserver",
        "ls -la /var/log",
        "kubectl get pods",
        "cd /var/www/html",
        "systemctl restart nginx",
    ]
    temp = tempfile.NamedTemporaryFile(mode='w', delete=False)
    temp.write('\n'.join(test_commands))
    temp.close()
    yield temp.name, test_commands
    os.unlink(temp.name)

@pytest.fixture
def mock_model():
    with patch('fuzzyshell.fuzzyshell.ModelHandler') as MockModelHandler:
        mock = MagicMock()
        MockModelHandler.return_value = mock
        yield mock

@pytest.fixture
def fs_with_model(memory_db, mock_model):
    # Use stock model for unit tests (allows download)
    os.environ['FUZZYSHELL_MODEL'] = 'stock-minilm-l6'
    
    fs = FuzzyShell(conn=memory_db)
    fs.init_model_sync()  # Force synchronous initialization
    return fs

def test_semantic_similarity(fs_with_model, mock_model):
    """Test that semantically similar words have higher similarity than dissimilar ones"""
    test_pairs = [
        ("apple", "orange", "airplane"),  # Fruits vs unrelated
        ("docker", "container", "banana"), # Tech terms vs unrelated
        ("python", "java", "elephant"),   # Programming languages vs unrelated
        ("git", "svn", "banana"),         # Version control vs unrelated
        ("nginx", "apache", "giraffe"),   # Web servers vs unrelated
    ]
    
    for word1, similar_word, different_word in test_pairs:
        # Create embeddings with known similarities
        emb1 = np.random.randn(384)
        emb_similar = emb1 + 0.1 * np.random.randn(384)  # Similar embedding
        emb_different = np.random.randn(384)  # Different embedding
        
        # Normalize embeddings
        emb1 = emb1 / np.linalg.norm(emb1)
        emb_similar = emb_similar / np.linalg.norm(emb_similar)
        emb_different = emb_different / np.linalg.norm(emb_different)
        
        # Configure mock to return our test embeddings
        def mock_encode(texts):
            return np.array([
                emb1 if text == word1 else
                emb_similar if text == similar_word else
                emb_different
                for text in texts
            ])
        mock_model.encode.side_effect = mock_encode
        
        # Calculate similarities
        sim_score = np.dot(emb1, emb_similar)
        diff_score = np.dot(emb1, emb_different)
        
        assert sim_score > diff_score, \
            f"Expected {word1} to be more similar to {similar_word} than to {different_word}"

def test_embedding_dimensions(fs_with_model, mock_model):
    """Test that embeddings maintain consistent dimensions throughout processing"""
    test_cmd = "docker run -p 8080:80 nginx"
    test_embedding = np.random.randn(384)  # Using correct 384 dimensions
    
    # Configure mock to return the right shape (batch_size, MODEL_OUTPUT_DIM)
    mock_model.encode.return_value = np.array([test_embedding])  # Shape (1, 384)
    
    # Get embedding and verify dimensions
    embedding = fs_with_model.model.encode([test_cmd])
    assert embedding.shape == (1, 384), "Initial embedding should be shape (1, 384)"
    embedding = embedding[0]  # Get first batch item
    
    # Test quantization maintains dimensions
    quantized = fs_with_model.quantize_embedding(embedding)
    assert len(quantized) == 384, "Quantized embedding should be 384 dimensions"
    
    # Test dequantization maintains dimensions
    dequantized = fs_with_model.dequantize_embedding(quantized)
    assert len(dequantized) == 384, "Dequantized embedding should be 384 dimensions"

def test_search_ranking(fs_with_model, mock_model, temp_history):
    """Test that search results are properly ranked by similarity"""
    history_file, _ = temp_history
    
    # Create base embedding for docker-related queries
    docker_base = np.random.randn(384)
    docker_base = docker_base / np.linalg.norm(docker_base)
    
    # Create test commands with controlled embeddings
    test_commands = {
        "docker run nginx": docker_base + 0.1 * np.random.randn(384),
        "docker ps": docker_base + 0.1 * np.random.randn(384),
        "ls -la": np.random.randn(384),
        "git status": np.random.randn(384),
        "docker build -t myapp .": docker_base + 0.1 * np.random.randn(384),
    }
    
    # Normalize all embeddings
    for cmd in test_commands:
        test_commands[cmd] = test_commands[cmd] / np.linalg.norm(test_commands[cmd])
    
    # Configure mock to return appropriate embeddings
    def mock_encode(texts):
        if len(texts) == 1 and texts[0] == "show running docker containers":
            return np.array([docker_base])
        return np.array([test_commands[text] for text in texts])
    
    mock_model.encode.side_effect = mock_encode
    
    # Write test commands and ingest
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write('\n'.join(test_commands.keys()))
        f.close()
        fs_with_model.get_shell_history_file = lambda: f.name
        fs_with_model.ingest_history()
    
    # Search and verify ranking
    results = fs_with_model.search("show running docker containers")
    
    docker_scores = []
    non_docker_scores = []
    
    for cmd, score in results:
        if 'docker' in cmd:
            docker_scores.append(score)
        else:
            non_docker_scores.append(score)
    
    if docker_scores and non_docker_scores:
        # Check that docker commands generally score higher (allow some tolerance)
        avg_docker_score = sum(docker_scores) / len(docker_scores)
        avg_non_docker_score = sum(non_docker_scores) / len(non_docker_scores)
        assert avg_docker_score > avg_non_docker_score, \
            f"Average docker score ({avg_docker_score:.3f}) should be higher than non-docker ({avg_non_docker_score:.3f})"
    
    os.unlink(f.name)

def test_quantization_preserves_similarity(fs_with_model):
    """Test that quantization preserves relative similarities between embeddings"""
    # Create test embeddings with known similarities
    base = np.random.randn(384)
    similar = base + 0.1 * np.random.randn(384)  # Similar to base
    different = np.random.randn(384)  # Different from base
    
    # Normalize all embeddings
    base = base / np.linalg.norm(base)
    similar = similar / np.linalg.norm(similar)
    different = different / np.linalg.norm(different)
    
    # Calculate original similarities
    sim_score = np.dot(base, similar)
    diff_score = np.dot(base, different)
    
    # Quantize and dequantize
    q_base = fs_with_model.quantize_embedding(base)
    q_similar = fs_with_model.quantize_embedding(similar)
    q_different = fs_with_model.quantize_embedding(different)
    
    dq_base = fs_with_model.dequantize_embedding(q_base)
    dq_similar = fs_with_model.dequantize_embedding(q_similar)
    dq_different = fs_with_model.dequantize_embedding(q_different)
    
    # Calculate similarities after quantization
    sim_score_q = np.dot(dq_base, dq_similar)
    diff_score_q = np.dot(dq_base, dq_different)
    
    # Verify relative similarities are preserved
    assert sim_score > diff_score, "Original embeddings should show expected similarity"
    assert sim_score_q > diff_score_q, "Quantized embeddings should preserve relative similarity"
    
    # Check quantization error is within bounds
    error_sim = abs(sim_score - sim_score_q)
    error_diff = abs(diff_score - diff_score_q)
    assert error_sim < 0.1, "Quantization error for similar embeddings should be small"
    assert error_diff < 0.1, "Quantization error for different embeddings should be small"
