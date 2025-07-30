from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from pathlib import Path
import time

def test_model_loading():
    print("\nTesting model loading and quantization...")
    cache_dir = Path.home() / '.cache' / 'fuzzyshell'
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / 'model.pt'

    # First time loading
    print("\n1. First time load:")
    start = time.time()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"Base model loaded in {time.time() - start:.2f}s")
    
    # Test basic encoding
    text = "test command"
    start = time.time()
    embedding = model.encode(text)
    print(f"Initial encoding took {time.time() - start:.2f}s")
    print(f"Embedding shape: {embedding.shape}")
    
    # Quantize
    print("\n2. Quantizing model:")
    start = time.time()
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module = torch.quantization.quantize_dynamic(
                module, {torch.nn.Linear}, dtype=torch.qint8
            )
    print(f"Quantization took {time.time() - start:.2f}s")
    
    # Test quantized encoding
    start = time.time()
    embedding = model.encode(text)
    print(f"Quantized encoding took {time.time() - start:.2f}s")
    
    # Save model
    print("\n3. Saving model:")
    start = time.time()
    torch.save(model.state_dict(), model_path)
    print(f"Model saved in {time.time() - start:.2f}s")
    print(f"Model file size: {model_path.stat().st_size / 1024 / 1024:.1f}MB")
    
    # Load saved model
    print("\n4. Loading from cache:")
    start = time.time()
    new_model = SentenceTransformer('all-MiniLM-L6-v2')
    new_model.load_state_dict(torch.load(model_path))
    print(f"Model loaded from cache in {time.time() - start:.2f}s")
    
    # Test cached model
    start = time.time()
    embedding = new_model.encode(text)
    print(f"Cached model encoding took {time.time() - start:.2f}s")
    
    return new_model

if __name__ == '__main__':
    model = test_model_loading()
