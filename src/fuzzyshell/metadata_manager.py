"""
Metadata Manager for FuzzyShell.

Centralizes metadata operations including database information gathering,
scoring preferences, model configuration, and system statistics.
"""

import logging
from typing import Dict, Any, Optional
from .data.datastore import MetadataDAL

logger = logging.getLogger('FuzzyShell.MetadataManager')


class MetadataManager:
    """
    Manages metadata operations for FuzzyShell.
    
    This class provides a high-level interface for:
    - Database information and statistics
    - Scoring preferences and configuration
    - Model metadata and version tracking
    - System state and diagnostics
    """
    
    def __init__(self, metadata_dal: MetadataDAL, fuzzyshell_instance=None):
        """
        Initialize MetadataManager.
        
        Args:
            metadata_dal: Data access layer for metadata operations
            fuzzyshell_instance: Optional FuzzyShell instance for accessing additional info
        """
        self.metadata_dal = metadata_dal
        self.fuzzyshell = fuzzyshell_instance
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get comprehensive database information.
        
        Returns:
            Dictionary containing database metadata, statistics, and configuration
        """
        info = {}
        
        try:
            # Get core database metadata
            db_metadata = self.metadata_dal.get_database_info()
            info.update(db_metadata)
            
            # Get additional statistics from FuzzyShell instance
            if self.fuzzyshell:
                try:
                    info['item_count'] = self.fuzzyshell.get_indexed_count()
                    info['embedding_model'] = self.metadata_dal.get('model_name') or 'unknown'
                    info['last_updated'] = self.metadata_dal.get('last_ann_build') or 'never'
                    
                    # Get database file path
                    if hasattr(self.fuzzyshell, 'db_path'):
                        info['db_path'] = self.fuzzyshell.db_path
                    elif hasattr(self.fuzzyshell, '_database_provider'):
                        info['db_path'] = getattr(self.fuzzyshell._database_provider, '_db_path', 'unknown')
                    else:
                        info['db_path'] = "~/.fuzzyshell/fuzzyshell.db"
                        
                except Exception as e:
                    logger.debug(f"Error getting FuzzyShell statistics: {e}")
                    
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            info = {'error': str(e)}
            
        return info
    
    def get_scoring_configuration(self) -> Dict[str, Any]:
        """
        Get current scoring configuration and preferences.
        
        Returns:
            Dictionary with scoring preferences and weights
        """
        try:
            preference = self.metadata_dal.get_scoring_preference()
            semantic_weight, bm25_weight = self.metadata_dal.get_scoring_weights()
            
            return {
                'preference': preference,
                'semantic_weight': semantic_weight,
                'bm25_weight': bm25_weight,
                'available_preferences': ['less_semantic', 'balanced', 'more_semantic', 'semantic_only']
            }
        except Exception as e:
            logger.error(f"Error getting scoring configuration: {e}")
            return {
                'preference': 'balanced',
                'semantic_weight': 0.5,
                'bm25_weight': 0.5,
                'error': str(e)
            }
    
    def set_scoring_preference(self, preference: str) -> bool:
        """
        Set scoring preference with validation.
        
        Args:
            preference: Scoring preference ('less_semantic', 'balanced', 'more_semantic', 'semantic_only')
            
        Returns:
            True if successfully set, False if error occurred
        """
        try:
            self.metadata_dal.set_scoring_preference(preference)
            logger.info(f"Updated scoring preference to: {preference}")
            return True
        except ValueError as e:
            logger.error(f"Invalid scoring preference '{preference}': {e}")
            return False
        except Exception as e:
            logger.error(f"Error setting scoring preference: {e}")
            return False
    
    def get_model_configuration(self) -> Dict[str, Any]:
        """
        Get current model configuration and metadata.
        
        Returns:
            Dictionary with model information
        """
        try:
            return {
                'model_name': self.metadata_dal.get('model_name') or 'unknown',
                'model_repo': self.metadata_dal.get('model_repo') or 'unknown',
                'model_dimensions': self.metadata_dal.get('model_dimensions') or 'unknown',
                'model_tokenizer_type': self.metadata_dal.get('model_tokenizer_type') or 'unknown',
                'model_key': self.metadata_dal.get('model_key') or 'unknown'
            }
        except Exception as e:
            logger.error(f"Error getting model configuration: {e}")
            return {'error': str(e)}
    
    def update_model_metadata(self, model_key: str, config: Dict[str, Any]) -> bool:
        """
        Update model metadata when model changes.
        
        Args:
            model_key: The model identifier key
            config: Model configuration dictionary
            
        Returns:
            True if successfully updated, False if error occurred
        """
        try:
            self.metadata_dal.set('model_name', model_key)
            self.metadata_dal.set('model_repo', config.get('repo', 'unknown'))
            self.metadata_dal.set('model_dimensions', str(config.get('files', {}).get('dimensions', 'unknown')))
            self.metadata_dal.set('model_tokenizer_type', config.get('files', {}).get('tokenizer_type', 'unknown'))
            self.metadata_dal.set('model_key', model_key)
            
            logger.info(f"Updated model metadata for: {model_key}")
            return True
        except Exception as e:
            logger.error(f"Error updating model metadata: {e}")
            return False
    
    def get_system_diagnostics(self) -> Dict[str, Any]:
        """
        Get system diagnostic information.
        
        Returns:
            Dictionary with system health and performance metrics
        """
        diagnostics = {
            'database_healthy': True,
            'model_loaded': False,
            'search_engine_ready': False,
        }
        
        try:
            # Check database health
            db_info = self.get_database_info()
            diagnostics['database_healthy'] = 'error' not in db_info
            diagnostics['item_count'] = db_info.get('item_count', 0)
            
            # Check FuzzyShell instance status
            if self.fuzzyshell:
                diagnostics['model_loaded'] = hasattr(self.fuzzyshell, '_model') and self.fuzzyshell._model is not None
                diagnostics['search_engine_ready'] = hasattr(self.fuzzyshell, 'search_engine') and self.fuzzyshell.search_engine is not None
                
                # Get embedding statistics
                try:
                    diagnostics['embedding_count'] = self.fuzzyshell.command_dal.get_embedding_count()
                    diagnostics['total_commands'] = self.fuzzyshell.command_dal.get_command_count()
                except Exception as e:
                    logger.debug(f"Error getting command statistics: {e}")
                    
        except Exception as e:
            logger.error(f"Error getting system diagnostics: {e}")
            diagnostics['error'] = str(e)
            
        return diagnostics
    
    def get_expert_screen_data(self) -> Dict[str, Any]:
        """
        Get comprehensive data for expert/debug screens.
        
        Returns:
            Dictionary with all technical information for expert display
        """
        try:
            # Combine all metadata sources
            db_info = self.get_database_info()
            scoring_config = self.get_scoring_configuration()
            model_config = self.get_model_configuration()
            diagnostics = self.get_system_diagnostics()
            
            # Import constants for display
            try:
                from .fuzzyshell import USE_ANN_SEARCH, ANN_NUM_CLUSTERS, ANN_CLUSTER_CANDIDATES, EMBEDDING_DTYPE, MODEL_OUTPUT_DIM
                ann_config = {
                    'use_ann_search': USE_ANN_SEARCH,
                    'num_clusters': ANN_NUM_CLUSTERS,
                    'cluster_candidates': ANN_CLUSTER_CANDIDATES,
                    'embedding_dtype': EMBEDDING_DTYPE.__name__ if hasattr(EMBEDDING_DTYPE, '__name__') else str(EMBEDDING_DTYPE),
                    'model_output_dim': MODEL_OUTPUT_DIM
                }
            except ImportError:
                ann_config = {
                    'use_ann_search': True,
                    'num_clusters': 32,
                    'cluster_candidates': 4,
                    'embedding_dtype': 'float32',
                    'model_output_dim': 384
                }
            
            return {
                'database': db_info,
                'scoring': scoring_config,
                'model': model_config,
                'diagnostics': diagnostics,
                'ann_config': ann_config,
                'version': self._get_version()
            }
            
        except Exception as e:
            logger.error(f"Error getting expert screen data: {e}")
            return {'error': str(e)}
    
    def _get_version(self) -> str:
        """Get FuzzyShell version."""
        try:
            from . import __version__
            return __version__
        except ImportError:
            return "0.1.0"
    
    def set_metadata(self, key: str, value: str) -> bool:
        """
        Set arbitrary metadata value.
        
        Args:
            key: Metadata key
            value: Metadata value
            
        Returns:
            True if successfully set, False if error occurred
        """
        try:
            self.metadata_dal.set(key, value)
            return True
        except Exception as e:
            logger.error(f"Error setting metadata {key}={value}: {e}")
            return False
    
    def get_metadata(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get arbitrary metadata value.
        
        Args:
            key: Metadata key
            default: Default value if key not found
            
        Returns:
            Metadata value or default
        """
        try:
            result = self.metadata_dal.get(key)
            return result if result is not None else default
        except Exception as e:
            logger.error(f"Error getting metadata {key}: {e}")
            return default