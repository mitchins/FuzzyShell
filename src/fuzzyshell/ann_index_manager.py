"""
ANN Index Manager - Handles Approximate Nearest Neighbor search indexing.

Extracted from the main FuzzyShell class for better testability and separation of concerns.
"""
import os
import pickle
import time
import logging
import numpy as np
from typing import List, Tuple, Optional, Callable

logger = logging.getLogger('FuzzyShell.ANNIndexManager')

class ANNSearchIndex:
    """Simple ANN search index using K-means clustering."""
    
    def __init__(self):
        self.cluster_centers = None
        self.cluster_indices = None  # Which commands belong to each cluster
        self.embeddings = None
        self.n_clusters = 0
        self.is_trained = False
    
    def fit(self, embeddings: np.ndarray, n_clusters: int = 32):
        """Train the ANN index on embeddings."""
        if len(embeddings) < n_clusters:
            logger.warning(f"Too few embeddings ({len(embeddings)}) for {n_clusters} clusters")
            return False
            
        self.n_clusters = n_clusters
        self.embeddings = embeddings.copy()
        
        # Simple K-means clustering
        self.cluster_centers, cluster_labels = self._kmeans(embeddings, n_clusters)
        
        # Build cluster indices (which embeddings belong to each cluster)
        self.cluster_indices = [[] for _ in range(n_clusters)]
        for embedding_idx, cluster_id in enumerate(cluster_labels):
            self.cluster_indices[cluster_id].append(embedding_idx)
        
        self.is_trained = True
        logger.info(f"ANN index trained with {len(embeddings)} embeddings, {n_clusters} clusters")
        return True
    
    def search(self, query_embedding: np.ndarray, n_candidates: int = 12) -> List[int]:
        """Search for candidate indices using ANN."""
        if not self.is_trained:
            logger.warning("ANN index not trained")
            return list(range(len(self.embeddings))) if self.embeddings is not None else []
        
        # Find closest cluster centers
        similarities = np.dot(self.cluster_centers, query_embedding)
        closest_clusters = np.argsort(similarities)[-n_candidates:]
        
        # Collect all candidates from closest clusters
        candidate_indices = []
        for cluster_id in closest_clusters:
            candidate_indices.extend(self.cluster_indices[cluster_id])
        
        return candidate_indices
    
    def _kmeans(self, embeddings: np.ndarray, k: int, max_iters: int = 50, tolerance: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
        """Simple K-means clustering implementation."""
        n_samples, n_features = embeddings.shape
        
        # Initialize centroids randomly
        np.random.seed(42)  # For reproducible results
        centroids = embeddings[np.random.choice(n_samples, k, replace=False)].copy()
        
        for iteration in range(max_iters):
            # Assign each point to closest centroid
            distances = np.dot(embeddings, centroids.T)  # Cosine similarity
            labels = np.argmax(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                cluster_points = embeddings[labels == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = np.mean(cluster_points, axis=0)
                    # Normalize for cosine similarity
                    new_centroids[i] = new_centroids[i] / np.linalg.norm(new_centroids[i])
                else:
                    # Keep old centroid if no points assigned
                    new_centroids[i] = centroids[i]
            
            # Check for convergence
            centroid_shift = np.mean(np.linalg.norm(new_centroids - centroids, axis=1))
            if centroid_shift < tolerance:
                logger.debug(f"K-means converged after {iteration + 1} iterations")
                break
                
            centroids = new_centroids
        
        # Final assignment
        distances = np.dot(embeddings, centroids.T)
        labels = np.argmax(distances, axis=1)
        
        return centroids, labels


class ANNIndexManager:
    """Manages ANN index persistence and lifecycle."""
    
    def __init__(self, db_path: str, command_dal=None, dequantize_func: Optional[Callable] = None,
                 embedding_dtype=None, embedding_dims: int = 384):
        """
        Initialize ANNIndexManager.
        
        Args:
            db_path: Path to database file
            command_dal: Data access layer for commands (optional, for rebuild functionality)
            dequantize_func: Function to dequantize embeddings (optional)
            embedding_dtype: Numpy dtype for embeddings (optional)
            embedding_dims: Number of embedding dimensions (default: 384)
        """
        self.db_path = db_path
        self.index = ANNSearchIndex()
        self._cache_path = None
        self.command_dal = command_dal
        self.dequantize_func = dequantize_func
        self.embedding_dtype = embedding_dtype or np.float32
        self.embedding_dims = embedding_dims
        
        if db_path and db_path != ':memory:' and db_path.endswith('.db'):
            self._cache_path = db_path.replace('.db', '_ann_index.pkl')
    
    def load_index(self, current_embedding_count: int = None) -> bool:
        """Load ANN index from disk cache."""
        if not self._cache_path:
            logger.debug("No cache path available for ANN index")
            return False
            
        if not os.path.exists(self._cache_path):
            logger.debug(f"ANN index cache does not exist: {self._cache_path}")
            return False
            
        try:
            with open(self._cache_path, 'rb') as f:
                data = pickle.load(f)
            
            # Validate data structure (indexed_count is optional for backward compatibility)
            required_keys = ['cluster_centers', 'cluster_indices', 'embeddings', 'n_clusters', 'is_trained']
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"Invalid cache: missing key '{key}'")
            
            # Check if the number of embeddings has changed significantly
            cached_count = data.get('indexed_count', len(data['embeddings']) if data['embeddings'] is not None else 0)
            if current_embedding_count is not None and cached_count > 0:
                count_diff = abs(current_embedding_count - cached_count)
                threshold = max(50, cached_count * 0.1)  # 50 commands or 10% change
                if count_diff > threshold:
                    logger.info(f"ANN index outdated: {count_diff} embedding count change ({cached_count} -> {current_embedding_count}, threshold: {threshold})")
                    return False
            
            # Restore index state
            self.index.cluster_centers = data['cluster_centers']
            self.index.cluster_indices = data['cluster_indices']
            self.index.embeddings = data['embeddings']
            self.index.n_clusters = data['n_clusters']
            self.index.is_trained = data['is_trained']
            
            logger.info(f"ANN index loaded from cache ({self._cache_path}) - {cached_count} embeddings indexed")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load ANN index: {e}")
            return False
    
    def save_index(self, embedding_count: int = None) -> bool:
        """Save ANN index to disk cache."""
        if not self._cache_path:
            logger.debug("No cache path available, skipping ANN index save")
            return False
            
        if not self.index.is_trained:
            logger.debug("ANN index not trained, skipping save")
            return False
        
        try:
            # Ensure parent directory exists
            parent_dir = os.path.dirname(self._cache_path)
            if not os.path.exists(parent_dir):
                raise RuntimeError(f"Database directory does not exist: {parent_dir}")
            
            # Use provided count or infer from embeddings
            indexed_count = embedding_count or (len(self.index.embeddings) if self.index.embeddings is not None else 0)
            
            data = {
                'cluster_centers': self.index.cluster_centers,
                'cluster_indices': self.index.cluster_indices,
                'embeddings': self.index.embeddings,
                'n_clusters': self.index.n_clusters,
                'is_trained': self.index.is_trained,
                'indexed_count': indexed_count,
                'created_timestamp': time.time()
            }
            
            with open(self._cache_path, 'wb') as f:
                pickle.dump(data, f)
                
            logger.info(f"ANN index saved to {self._cache_path} - {indexed_count} embeddings indexed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save ANN index: {e}")
            return False
    
    def build_index(self, embeddings: np.ndarray, n_clusters: int = 32) -> bool:
        """Build ANN index from embeddings."""
        start_time = time.time()
        
        if len(embeddings) < n_clusters:
            logger.info(f"Too few embeddings ({len(embeddings)}) for ANN clustering with {n_clusters} clusters")
            return False
        
        logger.info(f"Building ANN index with {len(embeddings)} embeddings, {n_clusters} clusters")
        
        success = self.index.fit(embeddings, n_clusters)
        
        if success:
            build_time = time.time() - start_time
            logger.info(f"ANN index built in {build_time:.3f}s")
            
            # Save to cache with embedding count
            self.save_index(len(embeddings))
        
        return success
    
    def rebuild_from_database(self, n_clusters: int = 32) -> bool:
        """
        Rebuild ANN index by loading embeddings from database.
        This encapsulates all the boilerplate of extracting and preparing embeddings.
        
        Args:
            n_clusters: Number of clusters for K-means
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.command_dal:
            logger.error("Cannot rebuild from database: command_dal not provided")
            return False
            
        try:
            # Load all embeddings from database
            all_data = self.command_dal.get_all_commands_with_embeddings_for_clustering()
            
            if len(all_data) < n_clusters:
                logger.info(f"Too few commands ({len(all_data)}) for ANN clustering with {n_clusters} clusters")
                return False
                
            # Extract and dequantize embeddings
            embeddings_list = []
            for _, _, emb_blob in all_data:
                # Convert blob to numpy array based on dtype
                if self.embedding_dtype == np.int8:
                    stored_emb = np.frombuffer(emb_blob, dtype=np.int8)[:self.embedding_dims]
                elif self.embedding_dtype == np.float16:
                    stored_emb = np.frombuffer(emb_blob, dtype=np.float16)[:self.embedding_dims]
                elif self.embedding_dtype == np.float32:
                    stored_emb = np.frombuffer(emb_blob, dtype=np.float32)[:self.embedding_dims]
                else:
                    raise ValueError(f"Unsupported embedding dtype: {self.embedding_dtype}")
                
                # Dequantize if function provided
                if self.dequantize_func:
                    embeddings_list.append(self.dequantize_func(stored_emb))
                else:
                    embeddings_list.append(stored_emb.astype(np.float32))
            
            embeddings = np.vstack(embeddings_list)
            
            # Build index
            success = self.build_index(embeddings, n_clusters)
            
            if success:
                logger.info(f"ANN index rebuilt from database with {len(embeddings)} embeddings")
            else:
                logger.error("Failed to rebuild ANN index from database")
                
            return success
            
        except Exception as e:
            logger.error(f"Error rebuilding ANN index from database: {e}")
            return False
    
    def search(self, query_embedding: np.ndarray, n_candidates: int = 12) -> List[int]:
        """Search for candidate indices."""
        return self.index.search(query_embedding, n_candidates)
    
    def is_trained(self) -> bool:
        """Check if index is trained and ready."""
        return self.index.is_trained
    
    def clear_cache(self):
        """Clear cached index file."""
        if self._cache_path and os.path.exists(self._cache_path):
            try:
                os.remove(self._cache_path)
                logger.info("ANN index cache cleared")
            except Exception as e:
                logger.warning(f"Failed to clear ANN cache: {e}")
        
        # Reset index
        self.index = ANNSearchIndex()
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        if not self.index.is_trained:
            return {'status': 'not_trained'}
        
        cluster_sizes = [len(indices) for indices in self.index.cluster_indices]
        
        return {
            'status': 'trained',
            'n_embeddings': len(self.index.embeddings),
            'n_clusters': self.index.n_clusters,
            'avg_cluster_size': np.mean(cluster_sizes),
            'min_cluster_size': np.min(cluster_sizes),
            'max_cluster_size': np.max(cluster_sizes),
            'empty_clusters': sum(1 for size in cluster_sizes if size == 0),
            'cache_path': self._cache_path
        }
    
    def check_health(self, current_embedding_count: int) -> dict:
        """
        Check ANN index health against current database state.
        
        Args:
            current_embedding_count: Current number of embeddings in database
            
        Returns:
            Dict with health status and recommendations
        """
        if not self.index.is_trained:
            return {
                'status': 'unhealthy',
                'reason': 'not_trained',
                'message': 'ANN index not trained',
                'staleness_pct': 0,
                'recommend_rebuild': True
            }
            
        # Get the count that was indexed
        indexed_count = len(self.index.embeddings) if self.index.embeddings is not None else 0
        
        if indexed_count == 0:
            return {
                'status': 'unhealthy', 
                'reason': 'empty_index',
                'message': 'ANN index is empty',
                'staleness_pct': 100,
                'recommend_rebuild': True
            }
        
        # Calculate staleness percentage
        if current_embedding_count >= indexed_count:
            staleness_pct = ((current_embedding_count - indexed_count) / indexed_count) * 100
        else:
            # Database shrunk - also concerning
            staleness_pct = ((indexed_count - current_embedding_count) / indexed_count) * 100
            
        # Health thresholds
        if staleness_pct >= 25:  # 25% or more change
            status = 'unhealthy'
            reason = 'very_stale'
            message = f'ANN index very stale: {staleness_pct:.0f}% change ({indexed_count} → {current_embedding_count})'
            recommend_rebuild = True
            
        elif staleness_pct >= 15:  # 15-25% change  
            status = 'degraded'
            reason = 'stale'
            message = f'ANN index stale: {staleness_pct:.0f}% change ({indexed_count} → {current_embedding_count})'
            recommend_rebuild = True
            
        elif staleness_pct >= 5:   # 5-15% change
            status = 'ok'
            reason = 'minor_staleness'
            message = f'ANN index slightly stale: {staleness_pct:.0f}% change'
            recommend_rebuild = False
            
        else:  # < 5% change
            status = 'healthy'
            reason = 'current'
            message = 'ANN index is current'
            recommend_rebuild = False
            
        return {
            'status': status,
            'reason': reason, 
            'message': message,
            'staleness_pct': round(staleness_pct, 1),
            'indexed_count': indexed_count,
            'current_count': current_embedding_count,
            'recommend_rebuild': recommend_rebuild
        }
        
    def get_health_warning(self, current_embedding_count: int) -> Optional[str]:
        """
        Get a simple warning message for display in UI if index needs attention.
        
        Args:
            current_embedding_count: Current number of embeddings in database
            
        Returns:
            Warning string for UI display, or None if healthy
        """
        health = self.check_health(current_embedding_count)
        
        if health['status'] == 'unhealthy':
            if health['reason'] == 'very_stale':
                return f"! Outdated Clusters ({health['staleness_pct']:.0f}% stale)"
            elif health['reason'] == 'not_trained':
                return "! No ANN Index"
            else:
                return "! ANN Index Issue"
                
        elif health['status'] == 'degraded':
            return f"⚠ Stale Clusters ({health['staleness_pct']:.0f}%)"
            
        return None  # Healthy or only minor staleness