import torch

# Suppress transformers warnings about missing PyTorch/TensorFlow
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='.*PyTorch.*TensorFlow.*Flax.*')
    from sentence_transformers import SentenceTransformer
import time
import numpy as np
from pathlib import Path
import sys

def get_model_size(model):
    """Get model size in MB"""
    total_size = 0
    for param in model.parameters():
        total_size += param.nelement() * param.element_size()
    return total_size / 1024 / 1024

def profile_model(model_name, precision="fp32", device="cpu"):
    print(f"\nTesting {model_name} ({precision} on {device}):")
    
    # Load model with timing breakdown
    model_download_start = time.time()
    model = SentenceTransformer(model_name)
    model_download_time = time.time() - model_download_start
    print(f"Download time: {model_download_time:.2f}s")
    
    # Get initial size
    initial_size = get_model_size(model)
    print(f"Initial size: {initial_size:.1f}MB")
    
    # Quantize if needed
    quantize_start = time.time()
    if precision == "fp16":
        model = model.half()
    elif precision == "int8":
        # Using dynamic quantization with detailed logging
        print("Starting INT8 quantization...")
        linear_count = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                linear_count += 1
                original_size = get_model_size(module)
                module = torch.quantization.quantize_dynamic(
                    module, {torch.nn.Linear}, dtype=torch.qint8
                )
                new_size = get_model_size(module)
                print(f"Layer {linear_count}: {original_size:.1f}MB -> {new_size:.1f}MB")
    
    quantize_time = time.time() - quantize_start
    print(f"Quantization time: {quantize_time:.2f}s")
    
    load_time = model_download_time + quantize_time
    print(f"Load time: {load_time:.2f}s")
    print(f"Model size: {get_model_size(model):.1f}MB")

    # Test data
    test_commands = [
        "git push origin main",
        "docker run -p 8080:80 nginx",
        "kubectl get pods --namespace default",
        "npm install --save-dev typescript",
        "python manage.py runserver 0.0.0.0:8000"
    ]

    test_queries = [
        "push to git",
        "start nginx container",
        "check kubernetes pods",
        "install typescript",
        "run django server"
    ]

    # Warmup with multiple iterations
    print("Warming up...")
    for _ in range(3):
        _ = model.encode("warmup text")
    
    # Measure encoding speed with multiple runs
    print("Measuring encoding speed...")
    runs = 5
    total_time = 0
    for run in range(runs):
        start_time = time.time()
        embeddings = model.encode(test_commands)
        run_time = (time.time() - start_time) / len(test_commands)
        total_time += run_time
        print(f"Run {run+1}: {run_time*1000:.1f}ms per command")
    
    encoding_time = total_time / runs
    print(f"Average encoding time per command: {encoding_time*1000:.1f}ms")

    # Measure similarity accuracy
    similarities = []
    for query, command in zip(test_queries, test_commands):
        query_embedding = model.encode(query)
        command_embedding = model.encode(command)
        similarity = torch.nn.functional.cosine_similarity(
            torch.tensor(query_embedding).unsqueeze(0),
            torch.tensor(command_embedding).unsqueeze(0)
        ).item()
        similarities.append(similarity)
    
    print(f"Average similarity score for matching pairs: {np.mean(similarities):.3f}")
    print(f"Memory usage: {torch.cuda.max_memory_allocated()/1024/1024:.1f}MB") if torch.cuda.is_available() else None

    return {
        'load_time': load_time,
        'model_size': get_model_size(model),
        'encoding_time': encoding_time,
        'similarity_score': np.mean(similarities)
    }

def main():
    models = [
        'all-MiniLM-L6-v2',
        'paraphrase-albert-small-v2',
        'paraphrase-MiniLM-L3-v2'
    ]
    
    precisions = ['fp32', 'fp16', 'int8']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    results = {}
    
    for model_name in models:
        results[model_name] = {}
        for precision in precisions:
            try:
                results[model_name][precision] = profile_model(model_name, precision, device)
            except Exception as e:
                print(f"Error testing {model_name} with {precision}: {e}")
    
    # Print comparison table
    print("\nComparison Table:")
    print("-" * 80)
    print(f"{'Model':<25} {'Precision':<8} {'Size (MB)':<10} {'Load (s)':<10} {'Encode (ms)':<12} {'Similarity':<10}")
    print("-" * 80)
    
    for model_name in results:
        for precision in results[model_name]:
            r = results[model_name][precision]
            print(f"{model_name[:25]:<25} {precision:<8} "
                  f"{r['model_size']:<10.1f} {r['load_time']:<10.2f} "
                  f"{r['encoding_time']*1000:<12.1f} {r['similarity_score']:<10.3f}")

if __name__ == '__main__':
    main()
