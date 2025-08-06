"""Expert screen controller for system information display."""

import logging
from typing import Dict, Any

logger = logging.getLogger('FuzzyShell.ExpertScreenController')

# Import configuration constants
try:
    from . import __version__
except ImportError:
    __version__ = "1.0.0"

try:
    from .fuzzyshell import USE_ANN_SEARCH, ANN_NUM_CLUSTERS, ANN_CLUSTER_CANDIDATES, EMBEDDING_DTYPE, MODEL_OUTPUT_DIM
except ImportError:
    # Fallback values for testing
    USE_ANN_SEARCH = True
    ANN_NUM_CLUSTERS = 32
    ANN_CLUSTER_CANDIDATES = 4
    EMBEDDING_DTYPE = type('MockDtype', (), {'__name__': 'float32'})()
    MODEL_OUTPUT_DIM = 384

try:
    from .model_configs import get_active_model_key, get_model_config
except ImportError:
    def get_active_model_key():
        return "mock-model"
    def get_model_config(key=None):
        return {"repo": "mock/model", "description": "Mock model for testing"}

try:
    from .model_handler import DescriptionHandler
except ImportError:
    DescriptionHandler = None


class ExpertScreenController:
    """
    Controller for expert screen data aggregation and display.
    
    This class centralizes the complex logic for gathering comprehensive
    system information from various FuzzyShell components. It provides
    a clean interface for expert/debug screens while handling errors
    gracefully and ensuring robust data collection.
    """
    
    def __init__(self, fuzzyshell_instance):
        """
        Initialize the expert screen controller.
        
        Args:
            fuzzyshell_instance: FuzzyShell instance to extract information from
        """
        self.fuzzyshell = fuzzyshell_instance
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information including models, database, and configuration.
        
        Returns:
            Dictionary with complete system information for expert display
        """
        info = {
            'version': __version__,
            'database': self._get_database_info(),
            'search_configuration': self._get_search_configuration(),
            'bm25_parameters': self._get_bm25_parameters(),
            'embedding_model': self._get_embedding_model_info(),
            'description_model': self._get_description_model_info(),
            'ann_index': self._get_ann_index_info()
        }
        
        return info
    
    def _get_database_info(self) -> Dict[str, Any]:
        """Get database information and statistics."""
        db_info = {
            'path': self.fuzzyshell.db_path,
            'total_commands': self.fuzzyshell.total_commands,
            'avg_command_length': self.fuzzyshell.avg_length,
        }
        
        # Get detailed database statistics
        try:
            command_count = self.fuzzyshell.command_dal.get_command_count()
            embedding_count = self.fuzzyshell.command_dal.get_embedding_count()
            cache_count = self.fuzzyshell.cache_dal.get_cache_count()
            
            db_info.update({
                'actual_command_count': command_count,
                'embedding_count': embedding_count,
                'embedding_coverage': f"{embedding_count/command_count*100:.1f}%" if command_count > 0 else "0%",
                'cached_queries': cache_count
            })
            
        except Exception as e:
            logger.error(f"Error getting database statistics: {e}")
            db_info['query_error'] = str(e)
        
        return db_info
    
    def _get_search_configuration(self) -> Dict[str, Any]:
        """Get search configuration information."""
        return {
            'use_ann_search': USE_ANN_SEARCH,
            'ann_num_clusters': ANN_NUM_CLUSTERS if USE_ANN_SEARCH else 'N/A',
            'ann_cluster_candidates': ANN_CLUSTER_CANDIDATES if USE_ANN_SEARCH else 'N/A',
            'embedding_dtype': EMBEDDING_DTYPE.__name__ if hasattr(EMBEDDING_DTYPE, '__name__') else str(EMBEDDING_DTYPE),
            'embedding_dimensions': MODEL_OUTPUT_DIM,
        }
    
    def _get_bm25_parameters(self) -> Dict[str, Any]:
        """Get BM25 search parameters."""
        return {
            'k1': self.fuzzyshell.k1,
            'b': self.fuzzyshell.b,
        }
    
    def _get_embedding_model_info(self) -> Dict[str, Any]:
        """Get embedding model information with fallback."""
        try:
            if self.fuzzyshell._model and hasattr(self.fuzzyshell._model, 'get_embedding_model_info'):
                return self.fuzzyshell._model.get_embedding_model_info()
            else:
                # Fallback when model not initialized
                current_model = get_active_model_key()
                config = get_model_config(current_model)
                return {
                    'status': 'Not initialized',
                    'model_key': current_model,
                    'model_name': config['repo'],
                    'description': config['description']
                }
        except Exception as e:
            logger.error(f"Error getting embedding model info: {e}")
            return {'error': str(e)}
    
    def _get_description_model_info(self) -> Dict[str, Any]:
        """Get description model information."""
        try:
            if DescriptionHandler is None:
                return {'error': 'DescriptionHandler not available'}
            
            desc_handler = DescriptionHandler()
            return desc_handler.get_model_info()
        except Exception as e:
            logger.error(f"Error getting description model info: {e}")
            return {'error': str(e)}
    
    def _get_ann_index_info(self) -> Dict[str, Any]:
        """Get ANN index status and statistics."""
        if USE_ANN_SEARCH and self.fuzzyshell.ann_manager:
            try:
                return self.fuzzyshell.ann_manager.get_stats()
            except Exception as e:
                logger.error(f"Error getting ANN index stats: {e}")
                return {'error': str(e)}
        else:
            return {'status': 'Disabled'}
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get comprehensive database information for status display."""
        # Get base database info from MetadataManager
        info = self.fuzzyshell.metadata_manager.get_database_info()
        
        # Add file-specific information
        db_size_bytes = self._get_database_size()
        db_size_human = self._format_bytes(db_size_bytes)
        
        info.update({
            'embedding_dtype': self.fuzzyshell.get_metadata('embedding_dtype', 'int8'),
            'schema_version': self.fuzzyshell.get_metadata('schema_version', '1.0'),
            'db_size_bytes': db_size_bytes,
            'db_size_human': db_size_human
        })
        
        return info
    
    def _get_database_size(self) -> int:
        """Get database size in bytes."""
        import os
        if os.path.exists(self.fuzzyshell.db_path):
            # For file-based databases
            return os.path.getsize(self.fuzzyshell.db_path)
        else:
            # For in-memory or non-existent databases, estimate based on record count
            return self.fuzzyshell.get_indexed_count() * 1024  # Rough estimate
    
    def _format_bytes(self, size_bytes: int) -> str:
        """Format bytes into human readable format."""
        if size_bytes < 1024:
            return f"{size_bytes}B"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.1f}KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes/(1024**2):.1f}MB"
        else:
            return f"{size_bytes/(1024**3):.1f}GB"