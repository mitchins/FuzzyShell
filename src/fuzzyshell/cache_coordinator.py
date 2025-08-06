"""
Cache Coordinator for FuzzyShell.

Coordinates various caching strategies including:
- Query result caching with hash-based keys
- ANN index cache management
- Cluster cache cleanup
"""

import os
import hashlib
import pickle
import logging
from typing import Optional, Any

logger = logging.getLogger('FuzzyShell.CacheCoordinator')


class CacheCoordinator:
    """Coordinates caching strategies for FuzzyShell."""
    
    def __init__(self, fuzzyshell_instance):
        """Initialize with reference to main FuzzyShell instance."""
        self.fuzzyshell = fuzzyshell_instance
        self.cache_dal = fuzzyshell_instance.cache_dal
        self.db_path = fuzzyshell_instance.db_path
        
    def cleanup_cache(self, max_age_hours: int = 24):
        """Clean up old cache entries."""
        self.cache_dal.cleanup_cache(max_age_hours)
    
    def clear_all_caches(self):
        """Clear all caches to force fresh results."""
        # Clear query cache
        self.cache_dal.clear_cache()
        logger.info("Cleared query cache")
        
        # Clear ANN cluster cache
        cluster_cache_path = self.db_path.replace('.db', '_clusters.pkl')
        if os.path.exists(cluster_cache_path):
            os.remove(cluster_cache_path)
            logger.info("Cleared ANN cluster cache")
            
        # Clear ANN index cache
        ann_cache_path = self.db_path.replace('.db', '_ann_index.pkl')
        if os.path.exists(ann_cache_path):
            os.remove(ann_cache_path)
            logger.info("Cleared ANN index cache")
            
        # Reset ANN manager cache if available
        if hasattr(self.fuzzyshell, 'ann_manager') and self.fuzzyshell.ann_manager:
            self.fuzzyshell.ann_manager.clear_cache()
            logger.info("Reset ANN index")
    
    def get_cached_results(self, query: str, return_scores: bool = False) -> Optional[Any]:
        """Get cached results for a query with specific return_scores setting."""
        cache_key = f"{query}|return_scores={return_scores}"
        query_hash = hashlib.md5(cache_key.encode()).hexdigest()
        
        cached_data = self.cache_dal.get_cached_results(query_hash, max_age_hours=1)
        if cached_data:
            return pickle.loads(cached_data)
        return None

    def cache_results(self, query: str, results: Any, return_scores: bool = False):
        """Cache results for a query with specific return_scores setting."""
        cache_key = f"{query}|return_scores={return_scores}"
        query_hash = hashlib.md5(cache_key.encode()).hexdigest()
        
        self.cache_dal.cache_query_results(query_hash, pickle.dumps(results))