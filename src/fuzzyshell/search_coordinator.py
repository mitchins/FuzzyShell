"""
Search Coordinator - Manages the complete search pipeline with proper separation of concerns.

Extracted from the monolithic FuzzyShell search method for better testability and debugging.
"""
import numpy as np
import logging
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger('FuzzyShell.SearchCoordinator')

@dataclass
class SearchResult:
    """Individual search result."""
    command: str
    semantic_score: float
    bm25_score: float
    combined_score: float
    rank: int

@dataclass
class SearchStats:
    """Search performance statistics."""
    total_time: float
    embedding_time: float
    ann_time: float
    similarity_time: float
    ranking_time: float
    candidates_considered: int
    results_returned: int

class EmbeddingManager:
    """Manages embedding generation, storage, and retrieval."""
    
    def __init__(self, model_handler, command_dal, dequantize_func, quantize_func=None):
        self.model = model_handler
        self.command_dal = command_dal
        self.dequantize_embedding = dequantize_func
        self.quantize_embedding = quantize_func
    
    def get_fresh_embedding(self, text: str) -> np.ndarray:
        """Get fresh embedding from model, quantized for fair comparison."""
        raw_embedding = self.model.encode([text])[0]
        # Apply same quantization process as stored embeddings for fair comparison
        if self.quantize_embedding and self.dequantize_embedding:
            quantized = self.quantize_embedding(raw_embedding)
            return self.dequantize_embedding(quantized)
        else:
            # Fallback: at least convert to FP32 to match stored embeddings
            return raw_embedding.astype(np.float32)
    
    def get_stored_embedding(self, command_id: int, stored_blob: bytes) -> np.ndarray:
        """Get stored embedding from database blob."""
        return self.dequantize_embedding(stored_blob)
    
    def verify_embedding_consistency(self, command: str, stored_embedding: np.ndarray) -> Tuple[bool, float]:
        """Verify stored embedding matches fresh model output."""
        fresh_embedding = self.get_fresh_embedding(command)
        consistency = np.dot(fresh_embedding, stored_embedding) / (
            np.linalg.norm(fresh_embedding) * np.linalg.norm(stored_embedding)
        )
        is_consistent = consistency > 0.95  # Should be very close to 1.0
        return is_consistent, consistency
    
    def diagnose_corruption(self, commands_with_embeddings: List[Tuple[int, str, bytes]]) -> Dict[str, Any]:
        """Diagnose embedding corruption in stored data."""
        corruption_report = {
            'total_tested': 0,
            'corrupted_count': 0,
            'corrupted_commands': [],
            'avg_consistency': 0.0,
            'min_consistency': 1.0,
            'max_consistency': -1.0
        }
        
        consistencies = []
        
        for cmd_id, command, emb_blob in commands_with_embeddings[:50]:  # Test first 50
            try:
                stored_emb = self.get_stored_embedding(cmd_id, emb_blob)
                is_consistent, consistency = self.verify_embedding_consistency(command, stored_emb)
                
                consistencies.append(consistency)
                corruption_report['total_tested'] += 1
                
                if not is_consistent:
                    corruption_report['corrupted_count'] += 1
                    corruption_report['corrupted_commands'].append({
                        'command': command,
                        'consistency': consistency
                    })
                
                corruption_report['min_consistency'] = min(corruption_report['min_consistency'], consistency)
                corruption_report['max_consistency'] = max(corruption_report['max_consistency'], consistency)
                
            except Exception as e:
                logger.error(f"Error testing embedding for '{command}': {e}")
                corruption_report['corrupted_count'] += 1
        
        if consistencies:
            corruption_report['avg_consistency'] = np.mean(consistencies)
        
        return corruption_report

class SimilarityCalculator:
    """Handles similarity calculations and ranking."""
    
    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    @staticmethod
    def batch_cosine_similarity(query_emb: np.ndarray, candidate_embs: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between query and batch of candidates."""
        # Normalize embeddings
        query_norm = query_emb / np.linalg.norm(query_emb)
        candidate_norms = candidate_embs / np.linalg.norm(candidate_embs, axis=1, keepdims=True)
        
        # Compute similarities
        similarities = np.dot(candidate_norms, query_norm)
        return similarities
    
    def rank_candidates(self, query_emb: np.ndarray, candidates: List[Tuple[int, str, np.ndarray]]) -> List[Tuple[int, str, float]]:
        """Rank candidates by similarity to query."""
        if not candidates:
            return []
        
        # Extract embeddings
        candidate_embs = np.vstack([emb for _, _, emb in candidates])
        
        # Calculate similarities
        similarities = self.batch_cosine_similarity(query_emb, candidate_embs)
        
        # Create ranked results
        ranked = []
        for i, (cmd_id, command, _) in enumerate(candidates):
            ranked.append((cmd_id, command, float(similarities[i])))
        
        # Sort by similarity descending
        ranked.sort(key=lambda x: x[2], reverse=True)
        
        return ranked

class SearchCoordinator:
    """Coordinates the complete search pipeline."""
    
    def __init__(self, model_handler, command_dal, ann_manager, dequantize_func, 
                 corpus_stats_dal=None, cache_dal=None, quantize_func=None):
        self.embedding_manager = EmbeddingManager(model_handler, command_dal, dequantize_func, quantize_func)
        self.similarity_calculator = SimilarityCalculator()
        self.command_dal = command_dal
        self.ann_manager = ann_manager
        self.corpus_stats_dal = corpus_stats_dal
        self.cache_dal = cache_dal
        
        # Configuration
        self.use_ann_search = ann_manager is not None
        self.ann_candidates = 12  # Default candidate count
        
    def search(self, query: str, top_k: int = 10) -> Tuple[List[SearchResult], SearchStats]:
        """Execute complete search pipeline."""
        start_time = time.time()
        
        # Stats tracking
        stats = SearchStats(
            total_time=0, embedding_time=0, ann_time=0, 
            similarity_time=0, ranking_time=0, 
            candidates_considered=0, results_returned=0
        )
        
        # Step 1: Generate query embedding
        embedding_start = time.time()
        query_embedding = self.embedding_manager.get_fresh_embedding(query)
        stats.embedding_time = time.time() - embedding_start
        
        # Step 2: Get candidate commands (ANN or full scan)
        ann_start = time.time()
        if self.use_ann_search and self.ann_manager and self.ann_manager.is_trained():
            candidate_indices = self.ann_manager.search(query_embedding, self.ann_candidates)
            logger.debug(f"ANN search returned {len(candidate_indices)} candidates")
        else:
            # Fallback to full scan
            candidate_indices = None
            logger.debug("Using full scan (ANN unavailable)")
        stats.ann_time = time.time() - ann_start
        
        # Step 3: Load candidate embeddings and commands
        similarity_start = time.time()
        if candidate_indices:
            # Load specific candidates
            candidates = self._load_candidate_embeddings(candidate_indices)
        else:
            # Load all embeddings
            candidates = self._load_all_embeddings()
        
        stats.candidates_considered = len(candidates)
        
        # Step 4: Calculate similarities and rank
        ranked_candidates = self.similarity_calculator.rank_candidates(query_embedding, candidates)
        stats.similarity_time = time.time() - similarity_start
        
        # Step 5: Apply final ranking and filtering
        ranking_start = time.time()
        search_results = self._create_search_results(ranked_candidates[:top_k], query)
        stats.ranking_time = time.time() - ranking_start
        
        stats.results_returned = len(search_results)
        stats.total_time = time.time() - start_time
        
        logger.info(f"Search completed in {stats.total_time:.3f}s: {len(search_results)} results")
        
        return search_results, stats
    
    def _load_candidate_embeddings(self, candidate_indices: List[int]) -> List[Tuple[int, str, np.ndarray]]:
        """Load embeddings for specific candidate indices."""
        candidates = []
        
        # Get all commands with embeddings (this is the current interface)
        all_data = self.command_dal.get_all_commands_with_embeddings_for_clustering()
        
        for i, (cmd_id, command, emb_blob) in enumerate(all_data):
            if i in candidate_indices:
                try:
                    embedding = self.embedding_manager.get_stored_embedding(cmd_id, emb_blob)
                    candidates.append((cmd_id, command, embedding))
                except Exception as e:
                    logger.warning(f"Failed to load embedding for '{command}': {e}")
        
        return candidates
    
    def _load_all_embeddings(self) -> List[Tuple[int, str, np.ndarray]]:
        """Load all embeddings for full scan."""
        candidates = []
        all_data = self.command_dal.get_all_commands_with_embeddings_for_clustering()
        
        for cmd_id, command, emb_blob in all_data:
            try:
                embedding = self.embedding_manager.get_stored_embedding(cmd_id, emb_blob)
                candidates.append((cmd_id, command, embedding))
            except Exception as e:
                logger.warning(f"Failed to load embedding for '{command}': {e}")
        
        return candidates
    
    def _create_search_results(self, ranked_candidates: List[Tuple[int, str, float]], query: str) -> List[SearchResult]:
        """Create SearchResult objects from ranked candidates."""
        results = []
        
        for rank, (cmd_id, command, similarity) in enumerate(ranked_candidates):
            # For now, semantic score is the similarity, BM25 would be added later
            result = SearchResult(
                command=command,
                semantic_score=similarity,
                bm25_score=0.0,  # TODO: Implement BM25 scoring
                combined_score=similarity,  # TODO: Combine semantic + BM25
                rank=rank + 1
            )
            results.append(result)
        
        return results
    
    def diagnose_embeddings(self, sample_size: int = 50) -> Dict[str, Any]:
        """Diagnose embedding corruption in the database."""
        logger.info(f"Diagnosing embedding corruption (sample size: {sample_size})")
        
        # Get sample of stored embeddings
        all_data = self.command_dal.get_all_commands_with_embeddings_for_clustering()
        sample_data = all_data[:sample_size]
        
        # Run corruption analysis
        report = self.embedding_manager.diagnose_corruption(sample_data)
        
        logger.info(f"Corruption diagnosis: {report['corrupted_count']}/{report['total_tested']} corrupted " +
                   f"(avg consistency: {report['avg_consistency']:.3f})")
        
        return report