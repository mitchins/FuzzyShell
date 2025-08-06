"""
Search Engine for FuzzyShell.

Extracted from the main FuzzyShell monolith to provide clean separation
of search functionality including semantic search, BM25 scoring, and
hybrid scoring strategies.
"""

import time
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
from .bm25_scorer import BM25Scorer
from .error_handler_util import ErrorHandlerUtil

logger = logging.getLogger(__name__)


class SearchEngine:
    """
    Core search engine implementing hybrid semantic + BM25 search.
    
    Handles:
    - Semantic similarity search using embeddings
    - BM25 keyword scoring 
    - User-configurable hybrid scoring
    - Query caching and optimization
    - ANN (Approximate Nearest Neighbor) acceleration
    """
    
    def __init__(self, 
                 model_handler,
                 command_dal,
                 metadata_dal,
                 cache_dal,
                 ann_manager=None,
                 quantize_func=None,
                 dequantize_func=None,
                 tokenizer_func=None):
        """
        Initialize search engine with required dependencies.
        
        Args:
            model_handler: Handle for semantic embedding model
            command_dal: Data access for commands and embeddings
            metadata_dal: Data access for metadata including scoring preferences
            cache_dal: Data access for query caching
            ann_manager: Optional ANN index manager for fast semantic search
            quantize_func: Function to quantize embeddings for storage
            dequantize_func: Function to dequantize embeddings from storage
            tokenizer_func: Function to tokenize text for BM25
        """
        self.model_handler = model_handler
        self.command_dal = command_dal
        self.metadata_dal = metadata_dal
        self.cache_dal = cache_dal
        self.ann_manager = ann_manager
        self.quantize_func = quantize_func
        self.dequantize_func = dequantize_func
        self.tokenizer_func = tokenizer_func or self._default_tokenize
        
        # Initialize BM25 scorer
        self.bm25_scorer = BM25Scorer(command_dal, tokenizer_func)
        
        # Model configuration
        self.model_output_dim = 384  # Default dimension
        
    def search(self, query: str, top_k: int = 50, return_scores: bool = False, 
               progress_callback: Optional[Callable] = None) -> List[Tuple]:
        """
        Main search function with hybrid semantic and keyword search.
        
        Args:
            query: Search query string  
            top_k: Maximum number of results to return
            return_scores: Whether to return detailed scores
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of (command, score) tuples, optionally with detailed scores
        """
        start_time = time.time()
        logger.info("🔍 SEARCH STARTING: query='%s' (return_scores=%s)", query, return_scores)
        
        if not query or query.isspace():
            return []
            
        # Check cache first
        try:
            cache_start = time.time()
            cached_results = self._get_cached_results(query, return_scores)
            if cached_results is not None:
                logger.debug("Cache hit! Retrieved results in %.3fs", time.time() - cache_start)
                return cached_results[:top_k]
            logger.debug("Cache miss, took %.3fs to check", time.time() - cache_start)
        except Exception as e:
            logger.error("Cache lookup failed: %s", str(e))
            
        # Ensure model is ready
        if not self._is_model_ready():
            ErrorHandlerUtil.log_and_raise_initialization_error("semantic model", logger)
            
        # BM25 scorer handles its own corpus stats

        try:
            # Generate and quantize query embedding
            embed_start = time.time()
            query_embedding = self._generate_query_embedding(query)
            logger.debug("Generated query embedding in %.3fs (shape: %s)", 
                        time.time() - embed_start, query_embedding.shape)
        except Exception as e:
            ErrorHandlerUtil.log_and_raise_operation_error(
                operation_name="query embedding generation",
                details=str(e),
                logger_instance=logger,
                cause=e
            )
        
        # Get candidate commands for search
        candidate_data, is_filtered = self._get_search_candidates(query, progress_callback)
        if not candidate_data:
            logger.warning("No commands found in database - have you run --ingest?")
            return []

        # Calculate scores using numpy vectorization for performance
        calc_start = time.time()
        results = self._calculate_hybrid_scores(
            query, query_embedding, candidate_data, return_scores, progress_callback, is_filtered
        )
        
        # Sort and limit results
        final_results = self._rank_and_limit_results(query, results, top_k, progress_callback)
        
        # Cache results for future queries
        try:
            self._cache_results(query, final_results, return_scores)
        except Exception as e:
            logger.warning("Failed to cache results: %s", str(e))
        
        # Final progress callback
        if progress_callback:
            progress_callback(len(candidate_data), len(candidate_data), "Complete", final_results)
        
        logger.info("🎯 SEARCH COMPLETE: %d results in %.3fs (query='%s')", 
                   len(final_results), time.time() - start_time, query)
        
        return final_results
    
    def _is_model_ready(self) -> bool:
        """Check if the semantic model is ready for use."""
        return self.model_handler is not None
    
    def _generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate and quantize embedding for query."""
        query_embedding = self.model_handler.encode([query])[0]
        
        # Ensure query embedding is the right size before quantizing
        if len(query_embedding) > self.model_output_dim:
            logger.debug("Truncating query embedding from %d to %d dimensions", 
                       len(query_embedding), self.model_output_dim)
            query_embedding = query_embedding[:self.model_output_dim]
            
        if self.quantize_func:
            query_embedding = self.quantize_func(query_embedding)
            
        return query_embedding
    
    def _get_search_candidates(self, query: str, progress_callback: Optional[Callable] = None) -> Tuple[List[Tuple], bool]:
        """Get candidate commands for search, prioritizing exact matches.
        
        Returns:
            Tuple of (candidate_data, is_filtered) where is_filtered indicates
            if results were filtered (which disables ANN search)
        """
        # First try to find commands that contain the search term
        exact_matches = self.command_dal.get_commands_with_embeddings_matching(f"%{query}%")
        
        # If we have enough exact matches, use those; otherwise get broader set
        if len(exact_matches) >= 20:
            all_commands_data = exact_matches
            is_filtered = True  # ANN can't be used with filtered results
            logger.debug("Using %d exact matches for query '%s' (ANN disabled)", len(exact_matches), query)
        else:
            all_commands_data = self.command_dal.get_all_commands_with_embeddings_ordered()
            is_filtered = False  # Full dataset - ANN can be used
            logger.debug("Using broader search with %d commands (full dataset, ANN available)", len(all_commands_data))
            
        return all_commands_data, is_filtered
    
    def _calculate_hybrid_scores(self, query: str, query_embedding: np.ndarray, 
                                candidate_data: List[Tuple], return_scores: bool,
                                progress_callback: Optional[Callable] = None, 
                                is_filtered: bool = False) -> List[Tuple]:
        """Calculate hybrid semantic + BM25 scores for candidates."""
        query_terms = self.tokenizer_func(query)
        command_ids, commands, embeddings = zip(*candidate_data)
        
        total_records = len(command_ids)
        logger.debug("Processing %d commands for ranking (query terms: %s)", total_records, query_terms)
        
        # Progress callback: Starting processing
        if progress_callback:
            progress_callback(0, total_records, "Loading embeddings...", [])
        
        # Load and process embeddings
        candidate_embeddings = self._load_embeddings_batch(embeddings, progress_callback, total_records)
        
        # Use ANN search if available and not working with filtered data
        use_ann_search = False
        if (self.ann_manager and self.ann_manager.is_trained() and 
            len(candidate_embeddings) > 100 and not is_filtered):
            
            ann_candidate_indices = self.ann_manager.search(query_embedding)
            candidate_embeddings = candidate_embeddings[ann_candidate_indices]
            candidate_commands = [commands[i] for i in ann_candidate_indices]
            candidate_command_ids = [command_ids[i] for i in ann_candidate_indices]
            candidate_indices = ann_candidate_indices
            use_ann_search = True
            logger.debug("Using ANN search for %d candidates", len(ann_candidate_indices))
        elif is_filtered:
            logger.debug("ANN search disabled - using filtered candidate set (%d commands)", len(candidate_embeddings))
        
        if not use_ann_search:
            candidate_commands = list(commands)
            candidate_command_ids = list(command_ids)
            candidate_indices = list(range(len(candidate_embeddings)))
            logger.debug("Using linear search for %d embeddings", len(candidate_embeddings))
        
        # Vectorized semantic similarity calculation
        semantic_scores = self._calculate_semantic_scores(candidate_embeddings, query_embedding, progress_callback)
        
        # Calculate BM25 scores in batch
        bm25_scores = self._calculate_bm25_scores_batch(query_terms, candidate_command_ids, progress_callback)
        
        # Normalize BM25 scores for consistent reporting
        normalized_bm25_scores = self._normalize_bm25_scores(bm25_scores)
        
        # Combine scores using user preferences (uses normalized scores internally)
        combined_scores = self._combine_scores(
            semantic_scores, normalized_bm25_scores, candidate_commands, query, progress_callback
        )
        
        # Package results
        results = []
        for i, (command, combined_score) in enumerate(zip(candidate_commands, combined_scores)):
            if return_scores:
                results.append((
                    command, float(combined_score), 
                    float(semantic_scores[i]), float(normalized_bm25_scores[i])
                ))
            else:
                results.append((command, float(combined_score)))
                
        return results
    
    def _load_embeddings_batch(self, embeddings: Tuple, progress_callback: Optional[Callable], 
                              total_records: int) -> np.ndarray:
        """Load embeddings from storage format into numpy array."""
        # Process in batches to show progress for large datasets
        batch_size = 1000
        num_batches = (len(embeddings) + batch_size - 1) // batch_size
        embeddings_list = []
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(embeddings))
            batch_embeddings = embeddings[start_idx:end_idx]
            
            # Progress callback for batch processing
            if progress_callback and len(embeddings) > 500:
                progress_callback(end_idx, total_records, f"Loading embeddings batch {batch_idx + 1}/{num_batches}...", [])
            
            # Dequantize embeddings if function is available
            for emb_blob in batch_embeddings:
                if self.dequantize_func:
                    dequantized = self.dequantize_func(emb_blob)
                else:
                    # Fallback: assume float32 format
                    dequantized = np.frombuffer(emb_blob, dtype=np.float32)
                    
                embeddings_list.append(dequantized)
        
        return np.vstack(embeddings_list)
    
    def _calculate_semantic_scores(self, candidate_embeddings: np.ndarray, 
                                  query_embedding: np.ndarray,
                                  progress_callback: Optional[Callable] = None) -> np.ndarray:
        """Calculate semantic similarity scores."""
        # Vectorized semantic similarity calculation on candidates
        semantic_scores = np.dot(candidate_embeddings, query_embedding.T).reshape(-1)
        norms = np.linalg.norm(candidate_embeddings, axis=1) * np.linalg.norm(query_embedding)
        
        logger.debug("Semantic scoring completed (max score: %.3f)", 
                    np.max(semantic_scores) if len(semantic_scores) > 0 else 0.0)
        
        # Progress callback: Semantic scoring complete
        if progress_callback:
            progress_callback(len(candidate_embeddings), len(candidate_embeddings), "Computing BM25 scores...", [])
        
        # Avoid division by zero and handle NaN values
        semantic_scores = np.divide(
            semantic_scores, 
            norms, 
            out=np.zeros_like(semantic_scores), 
            where=norms != 0
        )
        
        # Additional NaN/inf cleanup after division
        semantic_scores = np.nan_to_num(semantic_scores, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Ensure scores are in [0, 1] range
        semantic_scores = (semantic_scores + 1) / 2  # Convert from [-1, 1] to [0, 1]
        semantic_scores = np.clip(semantic_scores, 0, 1)
        
        # Final check for any remaining invalid values
        semantic_scores = np.nan_to_num(semantic_scores, nan=0.0, posinf=1.0, neginf=0.0)
        
        return semantic_scores
    
    def _calculate_bm25_scores_batch(self, query_terms: List[str], command_ids: List[int],
                                    progress_callback: Optional[Callable] = None) -> np.ndarray:
        """Calculate BM25 scores for a batch of commands."""
        return self.bm25_scorer.calculate_scores_batch(query_terms, command_ids)
    
    def _normalize_bm25_scores(self, bm25_scores: np.ndarray) -> np.ndarray:
        """Normalize BM25 scores to [0,1] range for consistent reporting."""
        if len(bm25_scores) == 0:
            return bm25_scores
            
        # Clean any NaN/inf values
        bm25_scores = np.nan_to_num(bm25_scores, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Normalize to [0,1] range  
        max_score = np.maximum(bm25_scores.max(), 1e-6)
        normalized = bm25_scores / max_score
        normalized = np.clip(normalized, 0, 1)
        
        # Final cleanup
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
        
        return normalized
    
    def _combine_scores(self, semantic_scores: np.ndarray, bm25_scores: np.ndarray,
                       commands: List[str], query: str,  
                       progress_callback: Optional[Callable] = None) -> np.ndarray:
        """Combine semantic and BM25 scores using user preferences."""
        if progress_callback:
            progress_callback(len(semantic_scores), len(semantic_scores), "Combining scores...", [])
        
        # BM25 scores are now pre-normalized, no need to normalize again
        
        # Dynamic hybrid scoring using user preferences
        combined_scores = []
        for i, (sem, bm25) in enumerate(zip(semantic_scores, bm25_scores)):
            hybrid = self._dynamic_hybrid_score(sem, bm25, commands[i], query)
            combined_scores.append(hybrid)
            
            # Log top scores for debugging
            if i < 5:
                logger.debug("Score: '%s' → Sem:%.3f BM25:%.3f Hybrid:%.3f", 
                           commands[i][:50], sem, bm25, hybrid)
        
        return np.array(combined_scores)
    
    def _dynamic_hybrid_score(self, semantic_score: float, bm25_score: float, 
                             command_text: str = "", query_text: str = "") -> float:
        """
        User-configurable hybrid scoring with rare term detection.
        
        Uses the user's scoring preference from metadata to balance semantic vs BM25 scores.
        Still applies rare term detection and phrase penalties for optimal results.
        """
        # Get user's scoring preference weights
        try:
            semantic_weight, bm25_weight = self.metadata_dal.get_scoring_weights()
        except:
            # Fallback to balanced if metadata unavailable
            semantic_weight, bm25_weight = 0.5, 0.5
        
        # Apply phrase proximity penalty to BM25 for multi-word queries
        if len(query_text.split()) > 1 and len(command_text.split()) > 1:
            original_bm25 = bm25_score
            bm25_score = self._apply_phrase_penalty(bm25_score, query_text, command_text)
            if abs(bm25_score - original_bm25) > 0.01:  # Log significant changes
                logger.debug("BM25 phrase penalty: '%s' vs '%s': %.3f -> %.3f", 
                           query_text, command_text, original_bm25, bm25_score)
        
        # Apply user's scoring preference with rare term adjustments
        has_rare_terms = self._query_has_rare_terms(query_text)
        
        # For semantic_only mode, ignore BM25 completely 
        if bm25_weight == 0.0:
            return semantic_score
        
        # For common terms with no special rarity, favor semantic search heavily
        if not has_rare_terms:
            # Almost eliminate BM25 influence for common terms (95% reduction)
            # This helps semantic search dominate for queries like "list files"
            adjusted_bm25_weight = bm25_weight * 0.05
            adjusted_semantic_weight = 1.0 - adjusted_bm25_weight
            return adjusted_semantic_weight * semantic_score + adjusted_bm25_weight * bm25_score
        
        # For rare terms, boost BM25 slightly (up to 20% increase)
        if semantic_score > 0.4 and bm25_score > 0.6:
            # Both scores are good - use base weights with small BM25 boost
            adjusted_bm25_weight = min(1.0, bm25_weight * 1.2)
            adjusted_semantic_weight = 1.0 - adjusted_bm25_weight
        else:
            # Use base weights for rare terms
            adjusted_semantic_weight, adjusted_bm25_weight = semantic_weight, bm25_weight
            
        return adjusted_semantic_weight * semantic_score + adjusted_bm25_weight * bm25_score
    
    def _apply_phrase_penalty(self, bm25_score: float, query_text: str, command_text: str) -> float:
        """Apply penalty to BM25 score for poor phrase matching in multi-word queries."""
        # This is a simplified version - the full implementation would be moved here
        return bm25_score * 0.9  # Simple 10% penalty for now
    
    def _query_has_rare_terms(self, query_text: str) -> bool:
        """Check if query contains rare terms that deserve BM25 boost."""
        # This is a placeholder - the actual rare term detection would be moved here
        common_terms = {'list', 'file', 'files', 'get', 'show', 'find', 'search', 'help', 'run', 'start', 'stop'}
        query_words = set(query_text.lower().split())
        return not query_words.issubset(common_terms)
    
    def _rank_and_limit_results(self, query: str, results: List[Tuple], top_k: int,
                               progress_callback: Optional[Callable] = None) -> List[Tuple]:
        """Sort results by score and apply exact match boosts."""
        def sort_key(result):
            command = result[0]
            score = result[1]
            
            # Only boost for exact command matches (not partial word matches)
            if command.lower() == query.lower():
                return score + 2.0  # Small boost for exact matches
            
            # Very small boost for exact prefix matches only
            if command.lower().startswith(query.lower()):
                return score + 0.1  # Minimal boost for prefix matches
            
            return score
        
        # Progress callback: Sorting results
        if progress_callback:
            progress_callback(len(results), len(results), "Sorting results...", [])
        
        results.sort(key=sort_key, reverse=True)
        return results[:top_k]
        
    def _get_cached_results(self, query: str, return_scores: bool) -> Optional[List[Tuple]]:
        """Get cached results for a query if available."""
        # This would implement caching logic
        return None
    
    def _cache_results(self, query: str, results: List[Tuple], return_scores: bool):
        """Cache results for future queries."""
        # This would implement result caching
        pass
    
    def invalidate_caches(self):
        """Invalidate search caches when corpus changes."""
        self.bm25_scorer.invalidate_cache()
        # Clear any query result caches here
    
    def _default_tokenize(self, text: str) -> List[str]:
        """Default tokenization - simple word splitting."""
        import re
        return re.findall(r'\b\w+\b', text.lower())