import sqlite3
import os
import sqlite_vss
import numpy as np
from collections import Counter
import math
import re
import asyncio
# threading import removed - no longer needed after async cleanup
import logging
import time
from pathlib import Path
import pickle
import hashlib
# threading.Event import removed - no longer needed after async cleanup
from .model_handler import ModelHandler, MODEL_OUTPUT_DIM
from .model_configs import get_active_model_key, get_model_config
from .fuzzy_tui import FuzzyShellApp, IngestionProgressTUI
from .data.datastore import CommandDAL, MetadataDAL, QueryCacheDAL, CorpusStatsDAL
from .data_processing import DataProcessingPipeline
from .ann_index_manager import ANNIndexManager
from .search_engine import SearchEngine
from .history_ingestion_coordinator import HistoryIngestionCoordinator
from .cache_coordinator import CacheCoordinator
from .embedding_processor import EmbeddingProcessor
from .tokenization_strategy import CommandTokenizationStrategy
from .expert_screen_controller import ExpertScreenController
from .startup_coordinator import StartupCoordinator
from .error_handler_util import ErrorHandlerUtil

# Configure logging
# File handler for debug logging
file_handler = logging.FileHandler('debug.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))

# Console handler for important messages only
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.CRITICAL)
console_handler.setFormatter(logging.Formatter('%(message)s'))

# Configure root logger
logger = logging.getLogger('FuzzyShell')
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Global flag to suppress print output during TUI mode
_TUI_MODE_ACTIVE = False

def set_tui_mode(active: bool):
    """Set TUI mode to suppress print statements."""
    global _TUI_MODE_ACTIVE
    _TUI_MODE_ACTIVE = active

def tui_safe_print(*args, **kwargs):
    """Print only if not in TUI mode."""
    if not _TUI_MODE_ACTIVE:
        print(*args, **kwargs)

# Configuration constants for embedding storage
EMBEDDING_DTYPE = np.float32  # Using FP32 for maximum precision - allows tweaking to FP16/INT8 later if needed
EMBEDDING_SCALE_FACTOR = 127  # Only used for INT8 quantization (currently unused with FP32)

# ANN Search Configuration
USE_ANN_SEARCH = True  # Enable K-means clustering for approximate nearest neighbor search
ANN_NUM_CLUSTERS = 32  # Number of clusters for K-means (tune based on dataset size)
ANN_CLUSTER_CANDIDATES = 12  # Number of closest clusters to search in (increased for better recall)


class FuzzyShell:
    def __init__(self, db_path=None, conn=None):
        # Set default database path if not provided
        if db_path is None:
            import os
            db_path = os.path.expanduser('~/.fuzzyshell/fuzzyshell.db')
        self.db_path = db_path
        self._model = None
        self._initialized = False
        
        # Initialize database provider
        if conn:
            # For tests: use MockDatabaseProvider with injected connection
            from .data.datastore import MockDatabaseProvider
            db_provider = MockDatabaseProvider(conn)
        else:
            # Production: use ProductionDatabaseProvider
            from .data.datastore import ProductionDatabaseProvider
            db_provider = ProductionDatabaseProvider(db_path)
        
        # Initialize all DALs with shared database provider
        from .data.datastore import CommandDAL, MetadataDAL, QueryCacheDAL, CorpusStatsDAL
        self.command_dal = CommandDAL(db_provider, load_vss=True)
        self.metadata_dal = MetadataDAL(db_provider)
        self.cache_dal = QueryCacheDAL(db_provider)
        self.stats_dal = CorpusStatsDAL(db_provider)
        
        # Initialize MetadataManager for high-level metadata operations
        from .metadata_manager import MetadataManager
        self.metadata_manager = MetadataManager(self.metadata_dal, self)
        
        # Initialize expert screen controller for system info
        self.expert_screen_controller = ExpertScreenController(self)
        
        # Initialize startup coordinator for model change management
        self.startup_coordinator = StartupCoordinator(self)
        
        # BM25 parameters
        self.k1 = 1.5
        self.b = 0.75
        
        # Cache for corpus stats
        self.total_commands = 0
        self.avg_length = 0
        
        # ANN search index manager - will be fully initialized after model is ready
        self.ann_manager = None
        
        # Initialize SearchEngine (will be configured when model is ready)
        self.search_engine = None
        
        # Data processing pipeline
        self.data_pipeline = None  # Will be initialized when model is ready
        
        # History ingestion coordinator
        self.history_ingestion_coordinator = HistoryIngestionCoordinator(self)
        
        # Cache coordinator
        self.cache_coordinator = CacheCoordinator(self)
        
        # Embedding processor
        self.embedding_processor = EmbeddingProcessor()
        
        # Tokenization strategy
        self.tokenization_strategy = CommandTokenizationStrategy()
        
        # ANN index initialization moved to init_model_sync after model is ready
        
        # Database initialization is handled by DAL during schema creation
    
    def rebuild_ann_index(self):
        """Rebuild ANN index - delegates to ANNIndexManager."""
        if not USE_ANN_SEARCH or not self.ann_manager:
            return False
            
        # Delegate all the work to the manager
        return self.ann_manager.rebuild_from_database(ANN_NUM_CLUSTERS)
    
    def init_model_sync(self):
        """Initialize the ONNX model handler synchronously"""
        start_time = time.time()
        logger.debug("[SYNC] Starting synchronous model initialization")
        
        try:
            logger.debug("[SYNC] Creating ModelHandler instance...")
            handler_start = time.time()
            
            self._model = ModelHandler()
            
            # Initialize data processing pipeline with model and ANN index
            # Create a DAL container for the pipeline
            class DALContainer:
                def __init__(self, command_dal):
                    self.command_dal = command_dal
                    
                def connection(self):
                    """Provide connection context manager compatible with pipeline commands."""
                    return self.command_dal.connection()
            
            dal_container = DALContainer(self.command_dal)
            
            self.data_pipeline = DataProcessingPipeline(
                dal_provider=dal_container,
                db_path=self.db_path,
                model_handler=self._model,
                ann_index_handler=self.ann_manager.index if self.ann_manager else None,
                tokenizer_func=self.tokenization_strategy.tokenize,
                embedding_dtype=EMBEDDING_DTYPE,
                model_output_dim=MODEL_OUTPUT_DIM,
                quantize_func=self.quantize_embedding  # Use FuzzyShell's quantization for compatibility
            )
            
            # Initialize ANN manager with all dependencies now that model is ready
            if USE_ANN_SEARCH:
                self.ann_manager = ANNIndexManager(
                    db_path=self.db_path,
                    command_dal=self.command_dal,
                    dequantize_func=self.dequantize_embedding,
                    embedding_dtype=EMBEDDING_DTYPE,
                    embedding_dims=MODEL_OUTPUT_DIM
                )
            
            # Initialize SearchEngine with all required dependencies
            self.search_engine = SearchEngine(
                model_handler=self._model,
                command_dal=self.command_dal,
                metadata_dal=self.metadata_dal,
                cache_dal=self.cache_dal,
                ann_manager=self.ann_manager,
                quantize_func=self.quantize_embedding,
                dequantize_func=self.dequantize_embedding,
                tokenizer_func=self.tokenization_strategy.tokenize
            )
            
            handler_time = time.time() - handler_start
            logger.debug(f"[SYNC] ModelHandler, pipeline, and SearchEngine created in {handler_time:.3f}s")
            
            # Check for model changes using DAL
            logger.debug("[SYNC] Checking for model changes...")
            self.startup_coordinator.handle_model_change_during_init()
            
            # Store current model metadata if not already stored
            metadata_model = self.metadata_dal.get('model_name')
            if not metadata_model:
                self.startup_coordinator.store_model_metadata()
            
            logger.debug("[SYNC] Checking for encode method...")
            if not hasattr(self._model, 'encode'):
                raise RuntimeError("Model initialized but encode method not found")
                
            total_time = time.time() - start_time
            # Initialize ANN index if needed
            if USE_ANN_SEARCH and self.ann_manager:
                # Get current embedding count to check if index is outdated
                try:
                    current_count = len(self.command_dal.get_all_commands_with_embeddings_for_clustering())
                except:
                    current_count = 0
                    
                # Try to load pre-trained index with embedding count check
                if not self.ann_manager.load_index(current_count):
                    if current_count == 0:
                        logger.info("Database is empty, skipping ANN index build for now")
                        # Only show message in CLI mode, not during healthy TUI startup
                        if not _TUI_MODE_ACTIVE:
                            tui_safe_print("⚠️  Database is empty, ANN index will be built after commands are added")
                    else:
                        logger.info("ANN index missing or outdated - rebuilding automatically...")
                        # Only show rebuild messages in CLI mode, not during healthy TUI startup
                        if not _TUI_MODE_ACTIVE:
                            tui_safe_print("🔧 ANN index missing or outdated - rebuilding automatically...")
                            tui_safe_print("⏳ This may take a few seconds for large command histories...")
                        
                        try:
                            # Use the new cleaner method
                            success = self.ann_manager.rebuild_from_database(ANN_NUM_CLUSTERS)
                            if success:
                                logger.info("ANN index rebuild complete")
                                if not _TUI_MODE_ACTIVE:
                                    tui_safe_print("✅ ANN index rebuild complete")
                            else:
                                logger.error("Failed to build ANN index during startup")
                                if not _TUI_MODE_ACTIVE:
                                    tui_safe_print("⚠️  ANN index rebuild failed, will retry later")
                        except Exception as e:
                            logger.error(f"ANN index rebuild failed: {e}")
                            if not _TUI_MODE_ACTIVE:
                                tui_safe_print("⚠️  ANN index rebuild failed, will retry later")
                else:
                    logger.info("ANN index loaded successfully from cache")
            
            logger.info(f"[SYNC] Model initialized successfully in {total_time:.3f}s")
            return self._model
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"[SYNC] Failed to initialize model after {total_time:.3f}s: %s", str(e), exc_info=True)
            raise
    
    def _store_model_metadata(self, model_key: str = None):
        """Store current model information in database metadata."""
        return self.startup_coordinator.store_model_metadata(model_key)
    
    def _handle_model_change(self):
        """Handle model change using DAL."""
        return self.startup_coordinator.handle_model_change_during_init()
            
    def quantize_embedding(self, embedding, scale_factor=EMBEDDING_SCALE_FACTOR):
        """Store embedding in the configured format (INT8, FP16, or FP32)."""
        return self.embedding_processor.quantize_embedding(embedding, scale_factor)
    
    def dequantize_embedding(self, stored_embedding, scale_factor=EMBEDDING_SCALE_FACTOR):
        """Convert stored embedding back to float32 for computation."""
        return self.embedding_processor.dequantize_embedding(stored_embedding, scale_factor)
    @property
    def model(self):
        """Get model, initializing synchronously if needed."""
        if self._model is None:
            logger.debug("Model access requested, initializing synchronously")
            try:
                self.init_model_sync()
            except Exception as e:
                logger.error("Model initialization failed: %s", str(e))
                return None
                
        return self._model
        
    def basic_search(self, query, top_k=50):
        """Fallback search using simple substring matching"""
        if not query or query.isspace():
            return []
            
        try:
            results = self.command_dal.basic_search(query, top_k)
            return results or [("No matches found", 0.0)]
            
        except Exception as e:
            tui_safe_print(f"Basic search failed: {str(e)}")
            return [("Search error occurred", 0.0)]
    
    
    def is_model_ready(self):
        """Check if model is ready without waiting"""
        return self._model is not None
    
    def get_indexed_count(self):
        """Get the total number of indexed commands from metadata (more accurate)."""
        try:
            return int(self.get_metadata('item_count', '0'))
        except Exception as e:
            logger.error("Error getting indexed count from metadata: %s", str(e))
            return self.command_dal.get_command_count()
    
    def get_metadata(self, key, default=None):
        """Get a metadata value by key."""
        return self.metadata_dal.get(key) or default
    
    def set_metadata(self, key, value):
        """Set a metadata key-value pair."""
        self.metadata_dal.set(key, value)
    
    
    
    def _update_item_count(self):
        """Update the item count in metadata to ensure accuracy."""
        try:
            count = self.command_dal.get_command_count()
            self.set_metadata('item_count', count)
            logger.debug("Updated item count to %d", count)
        except Exception as e:
            logger.error("Error updating item count: %s", str(e))
    
    def store_model_metadata(self, model_key: str = None):
        """Store current model information in database metadata."""
        return self.startup_coordinator.store_model_metadata(model_key)
    
    
    def detect_model_change(self) -> tuple[bool, str, str]:
        """
        Check if the active model has changed since last run.
        
        Returns:
            tuple: (changed, current_model, stored_model)
        """
        return self.startup_coordinator.detect_model_change()
    
    def handle_model_change(self):
        """Handle model change by clearing embeddings and updating metadata."""
        return self.startup_coordinator.handle_model_change()
    
    def clear_embeddings(self):
        """Clear all embeddings from the database."""
        try:
            self.command_dal.clear_embeddings()
            logger.info("Cleared all embeddings due to model change")
        except Exception as e:
            logger.error("Error clearing embeddings: %s", str(e))
    
    
    def get_database_info(self):
        """Get comprehensive database information for status display."""
        return self.expert_screen_controller.get_database_info()
    

    def add_command(self, raw_command):
        """Add a single command to the history and update its embedding."""
        if not raw_command or len(raw_command.strip()) == 0:
            return
        
        # Clean the command to remove shell history format
        command = self.clean_shell_command(raw_command)
        if not command or len(command.strip()) == 0:
            return
        
        try:
            # Prepare terms
            terms = command.lower().split()
            
            # Generate embedding if model is ready
            embedding = None
            if self._model is not None:
                embedding_vec = self.model.encode([command])[0]
                embedding_vec = embedding_vec[:384]  # Ensure correct dimensions
                embedding = self.quantize_embedding(embedding_vec)
            
            # Add command through DAL
            self.command_dal.add_single_command_with_terms(command, terms, embedding)
            
            # Update corpus stats
            self.stats_dal.increment_stats(len(terms))
            
            # Update item count metadata after successful addition
            self._update_item_count()
            
        except Exception as e:
            logger.error(f"Error adding command to database: {e}")

    def get_shell_history_file(self):
        if os.path.exists(".zsh_history"):
            return ".zsh_history"
        elif os.path.exists(".bash_history"):
            return ".bash_history"
        else:
            shell = os.environ.get("SHELL")
            if shell and "bash" in shell:
                return os.path.expanduser("~/.bash_history")
            elif shell and "zsh" in shell:
                return os.path.expanduser("~/.zsh_history")
            else:
                raise ValueError(f"Unsupported shell: {shell}. Only bash and zsh are currently supported.")

    def clean_shell_command(self, raw_command):
        """Clean shell history format to extract actual command"""
        # Handle zsh/bash history format: ': <timestamp>:<duration>;<command>'
        if raw_command.startswith(': ') and ';' in raw_command:
            # Find the semicolon that separates timestamp from command
            semicolon_pos = raw_command.find(';')
            if semicolon_pos != -1:
                # Extract command part after semicolon (no space expected after ;)
                command = raw_command[semicolon_pos + 1:].strip()
                return command if self._is_valid_shell_command(command) else None
        
        # If no shell history format detected, validate and return
        command = raw_command.strip()
        return command if self._is_valid_shell_command(command) else None
    
    def _is_valid_shell_command(self, command):
        """Check if a string is a valid shell command (not JSON or other data)"""
        if not command or command.isspace():
            return False
            
        # Filter out JSON fragments and data structures
        json_indicators = ['":', '": ', '",', '"]', '},', ':{', ':[', '\\\\']
        for indicator in json_indicators:
            if indicator in command:
                return False
                
        # Filter out lines that start with quotes (likely JSON keys/values)
        if command.strip().startswith('"') and command.strip().endswith('"'):
            return False
            
        return True


    def update_corpus_stats(self):
        """Update the average document length and total document count"""
        # Get counts and stats from DAL
        total_commands = self.command_dal.get_command_count()
        avg_length = self.command_dal.get_average_command_length()
        
        # Update stats through DAL
        self.stats_dal.update_stats(total_commands, avg_length)

        # Cache the values
        self.total_commands = total_commands
        self.avg_length = avg_length


    def ingest_history(self, use_tui=True, no_random=False):
        """Ingest shell history with optional TUI progress display."""
        return self.history_ingestion_coordinator.ingest_history(use_tui, no_random)


    def cleanup_cache(self, max_age_hours=24):
        """Clean up old cache entries."""
        return self.cache_coordinator.cleanup_cache(max_age_hours)
    
    def clear_all_caches(self):
        """Clear all caches to force fresh results."""
        return self.cache_coordinator.clear_all_caches()

    def get_cached_results(self, query, return_scores=False):
        """Get cached results for a query with specific return_scores setting."""
        return self.cache_coordinator.get_cached_results(query, return_scores)

    def cache_results(self, query, results, return_scores=False):
        """Cache results for a query with specific return_scores setting."""
        return self.cache_coordinator.cache_results(query, results, return_scores)

    def wait_for_model(self, timeout=10.0):
        """Wait for the model to be ready - using sync approach since model loads quickly"""
        logger.debug(f"[WAIT] wait_for_model called with timeout={timeout}")
        
        # Check if model and SearchEngine are already ready
        if (hasattr(self, '_model') and self._model is not None and 
            hasattr(self, 'search_engine') and self.search_engine is not None):
            logger.debug("[WAIT] Model and SearchEngine already loaded")
            return True
            
        # Model loads in ~0.3s, so just do it synchronously
        logger.debug("[WAIT] Loading model synchronously")
        try:
            start_time = time.time()
            result = self.init_model_sync()
            load_time = time.time() - start_time
            
            # Check that both model and SearchEngine were initialized
            if (result is not None and 
                hasattr(self, 'search_engine') and self.search_engine is not None):
                logger.info(f"[WAIT] Model and SearchEngine loaded successfully in {load_time:.3f}s")
                return True
            else:
                logger.error(f"[WAIT] Model or SearchEngine initialization failed after {load_time:.3f}s")
                logger.error(f"[WAIT] Model: {getattr(self, '_model', None) is not None}, SearchEngine: {getattr(self, 'search_engine', None) is not None}")
                return False
                
        except Exception as e:
            logger.error(f"[WAIT] Model initialization exception after {timeout}s timeout: {e}")
            return False

    def search(self, query, top_k=50, return_scores=False, progress_callback=None):
        """
        Main search function with hybrid semantic and keyword search.
        Now delegates to SearchEngine for clean separation of concerns.
        """
        # Ensure model and SearchEngine are ready
        model_ready = self.wait_for_model(timeout=5.0)
        if not model_ready:
            ErrorHandlerUtil.log_and_raise_initialization_error("semantic model", logger)
        
        if not self.search_engine:
            ErrorHandlerUtil.log_and_raise_initialization_error("search engine", logger)
        
        # Ensure corpus stats are updated for BM25 calculation
        if self.total_commands == 0:
            logger.debug("Updating corpus stats for BM25 scoring")
            self.update_corpus_stats()
        
        # Delegate to SearchEngine
        return self.search_engine.search(query, top_k, return_scores, progress_callback)
    
    def get_system_info(self) -> dict:
        """Get comprehensive system information including models, database, and configuration."""
        return self.expert_screen_controller.get_system_info()
    
    
    def tui(self):
        """Launch the interactive TUI for this FuzzyShell instance"""
        def search_callback(query: str) -> list:
            """Callback function for the UI to perform searches"""
            if not query:
                return []
            # Always return detailed scores for the TUI
            return self.search(query, return_scores=True)

        app = FuzzyShellApp(search_callback, fuzzyshell_instance=self)
        selected_command = app.run()
        return selected_command

__version__ = "1.0.0"


def interactive_search(show_profiling=False):
    """Launch the interactive search UI with onboarding if needed"""
    from .tui.simple_onboarding import check_needs_setup
    from .fuzzy_tui import FuzzyShellApp
    
    def create_and_run_tui(fuzzyshell_instance=None):
        """Create and run the main TUI with consistent setup."""
        # Create fresh instance if not provided
        fuzzyshell = fuzzyshell_instance or FuzzyShell()
        
        def search_callback(query: str) -> list:
            if not query:
                return []
            return fuzzyshell.search(query, return_scores=True)

        if show_profiling:
            tui_safe_print("🔍 Profiling mode enabled - check logs for detailed timing")
            
        app = FuzzyShellApp(search_callback, fuzzyshell_instance=fuzzyshell)
        return app.run()
    
    # Check for setup needs WITHOUT creating any instances
    if check_needs_setup():
        # First-time setup needed - run comprehensive onboarding
        from .tui.onboarding import run_comprehensive_onboarding
        return run_comprehensive_onboarding(create_and_run_tui, no_random=show_profiling)
    else:
        # No setup needed - run main TUI directly
        return create_and_run_tui()

def main():
    """Main entry point for the application"""
    from .cli_commands import main as cli_main
    cli_main()

if __name__ == '__main__':
    main()
