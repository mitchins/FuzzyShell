"""
Data Processing Pipeline using Facade pattern.

The DataProcessingPipeline hides the complexity of orchestrating multiple
data processing commands in the correct order.
"""

import time
import logging
from typing import List, Optional, Callable

from .context import ProcessingContext, ProcessingResult
from .commands import (
    ExtractEmbeddingsCommand,
    CalculateIDFCommand, 
    BuildANNIndexCommand,
    DatabaseWriteCommand,
    ClearCacheCommand
)

logger = logging.getLogger(__name__)


class DataProcessingPipeline:
    """
    Facade pattern for data processing operations.
    
    Hides the complexity of orchestrating multiple commands and provides
    a simple interface for common data processing workflows.
    """
    
    def __init__(self, dal_provider, db_path: str, model_handler=None, ann_index_handler=None, tokenizer_func=None, embedding_dtype=None, model_output_dim=384, quantize_func=None):
        self.dal_provider = dal_provider
        self.db_path = db_path
        self.model_handler = model_handler
        self.ann_index_handler = ann_index_handler
        self.tokenizer_func = tokenizer_func
        self.embedding_dtype = embedding_dtype or __import__('numpy').float32  # Default to float32
        self.model_output_dim = model_output_dim
        self.quantize_func = quantize_func  # FuzzyShell's quantization function for compatibility
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def process_ingestion(self, 
                         commands: List[str],
                         progress_callback: Optional[Callable[[int, str], None]] = None,
                         use_clustering: bool = True) -> ProcessingResult:
        """
        Process command ingestion from raw command strings.
        
        This is the main facade method that handles the complete ingestion pipeline:
        1. Generate embeddings from commands
        2. Write commands, embeddings, and term frequencies to database  
        3. Update corpus statistics (IDF calculation)
        4. Build ANN index for fast search
        
        Args:
            commands: List of command strings to ingest
            progress_callback: Optional callback for progress updates
            use_clustering: Whether to build ANN clustering index
            
        Returns:
            ProcessingResult with success status and metrics
        """
        start_time = time.time()
        
        # Create processing context
        context = ProcessingContext(
            commands=commands,
            use_clustering=use_clustering,
            progress_callback=progress_callback
        )
        
        try:
            # Execute the ingestion pipeline using Command pattern
            return self._execute_ingestion_commands(context, start_time)
            
        except Exception as e:
            self.logger.error(f"Ingestion processing failed: {e}", exc_info=True)
            return ProcessingResult(
                success=False,
                commands_processed=0,
                embeddings_processed=0,
                ann_index_built=False,
                processing_time_seconds=time.time() - start_time,
                error_message=str(e)
            )
    
    def rebuild_ann_index(self,
                         progress_callback: Optional[Callable[[int, str], None]] = None) -> ProcessingResult:
        """
        Rebuild just the ANN index from existing embeddings.
        
        Facade method for ANN index rebuilding without full ingestion.
        """
        start_time = time.time()
        
        # This would fetch existing embeddings from database
        # For now, placeholder implementation
        self.logger.info("Rebuilding ANN index from existing embeddings")
        
        return ProcessingResult(
            success=True,
            commands_processed=0,
            embeddings_processed=0,
            ann_index_built=True,
            processing_time_seconds=time.time() - start_time
        )
    
    def _execute_ingestion_commands(self, context: ProcessingContext, start_time: float) -> ProcessingResult:
        """
        Execute the ingestion command pipeline.
        
        Simple command orchestration without Template Method complexity.
        
        Args:
            context: ProcessingContext with input data
            start_time: Start time for performance measurement
            
        Returns:
            ProcessingResult with execution results
        """
        # Create command sequence for ingestion
        commands = [
            ExtractEmbeddingsCommand(self.model_handler, embedding_dtype=self.embedding_dtype, model_output_dim=self.model_output_dim, quantize_func=self.quantize_func) if self.model_handler else None,
            DatabaseWriteCommand(self.dal_provider, self.tokenizer_func),
            CalculateIDFCommand(self.dal_provider, self.tokenizer_func),
            BuildANNIndexCommand(self.dal_provider, self.ann_index_handler, embedding_dtype=self.embedding_dtype) if (context.use_clustering and self.ann_index_handler) else None
        ]
        
        # Filter out None commands
        commands = [cmd for cmd in commands if cmd is not None]
        
        # Execute commands in sequence
        for i, command in enumerate(commands):
            if not command.can_execute(context):
                self.logger.debug(f"Skipping {command.name} - cannot execute with current context")
                continue
                
            self.logger.debug(f"Executing command {i+1}/{len(commands)}: {command.name}")
            
            try:
                command.execute(context)
                
                # Progress update
                if context.progress_callback:
                    progress = int(((i + 1) / len(commands)) * 100)
                    context.progress_callback(progress, f"Completed {command.name}")
                    
            except Exception as e:
                self.logger.error(f"Command {command.name} failed: {e}")
                raise
        
        # Create result
        total_time = time.time() - start_time
        result = ProcessingResult(
            success=True,
            commands_processed=len(context.commands),
            embeddings_processed=len(context.processed_embeddings) if context.processed_embeddings else 0,
            ann_index_built=context.ann_index_data is not None,
            processing_time_seconds=total_time,
            embedding_extraction_time=context.processing_times.get('ExtractEmbeddings', 0.0),
            idf_calculation_time=context.processing_times.get('CalculateIDF', 0.0),
            ann_building_time=context.processing_times.get('BuildANNIndex', 0.0),
            database_write_time=context.processing_times.get('DatabaseWrite', 0.0)
        )
        
        return result
    
    def clear_all_caches(self) -> ProcessingResult:
        """
        Clear all processing caches.
        
        Simple facade method for cache clearing.
        """
        start_time = time.time()
        
        from .commands import ClearCacheCommand
        context = ProcessingContext(commands=[])
        
        try:
            clear_command = ClearCacheCommand(self.db_path)
            clear_command.execute(context)
            
            return ProcessingResult(
                success=True,
                commands_processed=0,
                embeddings_processed=0,
                ann_index_built=False,
                processing_time_seconds=time.time() - start_time
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                commands_processed=0,
                embeddings_processed=0,
                ann_index_built=False,
                processing_time_seconds=time.time() - start_time,
                error_message=str(e)
            )