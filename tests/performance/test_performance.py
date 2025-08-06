import time
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from fuzzyshell.fuzzyshell import FuzzyShell

def test_startup_and_search():
    print("\nTesting FuzzyShell Startup and Search Performance")
    print("=" * 50)
    
    # Test initial startup (cold)
    print("\n1. First time startup (expect model download & quantization):")
    start = time.time()
    fs = FuzzyShell()
    
    # Add some sample commands
    sample_commands = [
        "git push origin main",
        "docker run -p 80:80 nginx",
        "kubectl get pods --namespace default",
        "python manage.py runserver 0.0.0.0:8000",
        "npm install typescript --save-dev",
        "git commit -m 'Initial commit'",
        "docker-compose up -d",
        "kubectl apply -f deployment.yaml",
        "python -m pip install -r requirements.txt",
        "npm run build"
    ]
    
    for cmd in sample_commands:
        fs.add_command(cmd)
        
    print(f"Initial constructor time: {(time.time() - start)*1000:.1f}ms")
    
    # Test model initialization
    print("\n2. Testing model initialization and first query:")
    start = time.time()
    results = fs.search("git push", top_k=3)
    first_query = time.time() - start
    print(f"First query time (including model load): {first_query*1000:.1f}ms")
    print("\nSample results:")
    for cmd in results[:3]:
        print(f"- {cmd}")
    
    # Test cached performance
    print("\n3. Testing subsequent query performance:")
    times = []
    queries = [
        "docker run nginx",
        "git commit message",
        "kubernetes get pods",
        "python server",
        "npm install"
    ]
    
    for query in queries:
        start = time.time()
        results = fs.search(query, top_k=3)
        query_time = time.time() - start
        times.append(query_time)
        print(f"\nQuery: '{query}' ({query_time*1000:.1f}ms)")
        for cmd in results[:3]:
            print(f"- {cmd}")
    
    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    print(f"\nAverage query time: {avg_time:.1f}ms (±{std_time:.1f}ms)")
    
    # Test second startup (warm, from cache)
    print("\n4. Testing cached startup:")
    start = time.time()
    fs2 = FuzzyShell()
    results = fs2.search("test query", top_k=1)  # Force model load
    print(f"Cached startup time: {(time.time() - start)*1000:.1f}ms")

if __name__ == '__main__':
    test_startup_and_search()
