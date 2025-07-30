#!/usr/bin/env python3
"""
Hybrid BM25 + Semantic search example for FuzzyShell
Requires: sentence-transformers, rank_bm25, numpy
pip install sentence-transformers rank_bm25 numpy
"""

from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import numpy as np

# 1) Sample command history
commands = [
    "ls -lh",
    "git status",
    "docker ps -a",
    "ls --help",
    "python -m http.server",
    "grep -R 'TODO' .",
    "history | tail -n 50"
]

# 2) Tokenize commands for BM25
tokenized_cmds = [cmd.split() for cmd in commands]

# 3) Instantiate BM25
bm25 = BM25Okapi(tokenized_cmds)

# 4) Load your fine-tuned embedding model
embedder = SentenceTransformer(
    "Mitchins/minilm-l6-v2-terminal-describer-embeddings"
)

# 5) Precompute command embeddings (tensor)
command_embeddings = embedder.encode(
    commands,
    convert_to_tensor=True,
    normalize_embeddings=True
)

def hybrid_search(
    query: str,
    top_k: int = 5,
    alpha: float = 0.5  # blend factor: 0 = pure semantic, 1 = pure BM25
):
    # --- BM25 scores ---
    tokenized_query = query.split()
    raw_bm25 = np.array(bm25.get_scores(tokenized_query), dtype=np.float32)
    
    # Min–max normalize BM25 to [0,1]
    bm25_min, bm25_max = raw_bm25.min(), raw_bm25.max()
    norm_bm25 = (raw_bm25 - bm25_min) / (bm25_max - bm25_min + 1e-8)
    
    # --- Semantic scores ---
    # Embed query, normalize
    query_emb = embedder.encode(
        [query],
        convert_to_tensor=True,
        normalize_embeddings=True
    )
    raw_sem = util.cos_sim(query_emb, command_embeddings).cpu().numpy().flatten()
    
    # Min–max normalize semantic to [0,1]
    sem_min, sem_max = raw_sem.min(), raw_sem.max()
    norm_sem = (raw_sem - sem_min) / (sem_max - sem_min + 1e-8)
    
    # --- Hybrid blend ---
    hybrid_score = alpha * norm_bm25 + (1 - alpha) * norm_sem
    
    # --- Return top_k commands ---
    idx = np.argsort(hybrid_score)[::-1][:top_k]
    results = []
    for i in idx:
        results.append({
            "command": commands[i],
            "bm25": float(norm_bm25[i]),
            "sem": float(norm_sem[i]),
            "hybrid": float(hybrid_score[i])
        })
    return results

if __name__ == "__main__":
    for query in ["list files", "docker containers", "show HTTP server"]:
        print(f"\nQuery: {query!r}")
        for r in hybrid_search(query, top_k=3, alpha=0.5):
            print(f"  {r['command']:<20}  bm25={r['bm25']:.2f}  sem={r['sem']:.2f}  hybrid={r['hybrid']:.2f}")