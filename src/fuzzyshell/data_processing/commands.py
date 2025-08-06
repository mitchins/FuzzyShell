"""
Command Pattern implementations for data processing operations.

Each command encapsulates a specific data processing step and can be
executed independently or as part of a pipeline.
"""

import time
import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

from .context import ProcessingContext, ProcessingResult

logger = logging.getLogger(__name__)


class ProcessingCommand(ABC):
    """
    Abstract base class for data processing commands.
    
    Implements the Command pattern - each command encapsulates
    a specific operation that can be executed, undone, and logged.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.execution_time = 0.0
        
    @abstractmethod
    def execute(self, context: ProcessingContext) -> None:
        """
        Execute the command with the given context.
        
        Args:
            context: ProcessingContext containing input data and results
            
        Raises:
            Exception: If the command fails to execute
        """
        pass
    
    def can_execute(self, context: ProcessingContext) -> bool:
        """
        Check if this command can be executed with the given context.
        
        Args:
            context: ProcessingContext to check
            
        Returns:
            True if command can execute, False otherwise
        """
        return True
    
    def get_description(self) -> str:
        """Get human-readable description of this command."""
        return f"{self.name} processing step"


class ExtractEmbeddingsCommand(ProcessingCommand):
    """
    Command to extract embeddings from raw command data.
    
    Uses FuzzyShell's quantization methods for compatibility.
    """
    
    def __init__(self, model_handler, embedding_dtype=np.float32, model_output_dim=384, quantize_func=None):
        super().__init__("ExtractEmbeddings")
        self.model_handler = model_handler
        self.embedding_dtype = embedding_dtype
        self.model_output_dim = model_output_dim
        self.quantize_func = quantize_func  # Use FuzzyShell's quantization function
        
    def execute(self, context: ProcessingContext) -> None:
        """Generate embeddings from the command strings."""
        start_time = time.time()
        
        if not context.commands:
            raise ValueError("No commands provided in context")
            
        logger.debug(f"Generating embeddings for {len(context.commands)} commands")
        
        # Generate embeddings in batches for optimal performance
        BATCH_SIZE = 32  # Optimal batch size for best performance (256+ commands/sec)
        all_embeddings = []
        
        for batch_start in range(0, len(context.commands), BATCH_SIZE):
            batch_commands = context.commands[batch_start:batch_start + BATCH_SIZE]
            
            # Progress callback during batch processing
            if context.progress_callback:
                commands_processed = min(batch_start + BATCH_SIZE, len(context.commands))
                progress = 20 + int((batch_start / len(context.commands)) * 60)  # 20-80% range
                context.progress_callback(progress, f"Processing {commands_processed}/{len(context.commands)} commands")
            
            # Generate embeddings for this batch
            batch_embeddings = self.model_handler.encode(batch_commands)
            
            # Process each embedding in the batch
            for embedding in batch_embeddings:
                # Use FuzzyShell's quantization function if provided, otherwise fallback
                if self.quantize_func:
                    quantized_embedding = self.quantize_func(embedding)
                else:
                    # Fallback to simple quantization (for compatibility)
                    quantized_embedding = self._quantize_embedding_fallback(embedding)
                all_embeddings.append(quantized_embedding)
        
        context.processed_embeddings = all_embeddings
        self.execution_time = time.time() - start_time
        context.processing_times[self.name] = self.execution_time
        
        logger.debug(f"Generated {len(all_embeddings)} embeddings in {self.execution_time:.2f}s")
    
    def _quantize_embedding_fallback(self, embedding):
        """Fallback quantization method (for compatibility when quantize_func not provided)."""
        # Truncate to model dimensions if needed
        if len(embedding) > self.model_output_dim:
            embedding = embedding[:self.model_output_dim]
            
        if self.embedding_dtype == np.int8:
            # Scale to int8 range [-127, 127] and convert
            return (embedding * 127.0).astype(np.int8)
        elif self.embedding_dtype == np.float16:
            # Convert to FP16 for reduced storage
            return embedding.astype(np.float16)
        elif self.embedding_dtype == np.float32:
            # Store as FP32 for maximum precision (default)
            return embedding.astype(np.float32)
        else:
            raise ValueError(f"Unsupported embedding_dtype: {self.embedding_dtype}")
    
    def can_execute(self, context: ProcessingContext) -> bool:
        return context.commands is not None and len(context.commands) > 0


class CalculateIDFCommand(ProcessingCommand):
    """
    Command to calculate Inverse Document Frequency (IDF) values.
    
    IDF values are used for BM25 scoring in search operations.
    Updates corpus statistics for future search operations.
    """
    
    def __init__(self, dal_provider, tokenizer_func=None):
        super().__init__("CalculateIDF")
        self.dal_provider = dal_provider
        self.tokenizer_func = tokenizer_func or self._default_tokenize
        
    def execute(self, context: ProcessingContext) -> None:
        """Calculate IDF values and update corpus statistics."""
        start_time = time.time()
        
        logger.debug("Calculating IDF values and updating corpus statistics")
        
        # Update corpus stats using the DAL - this recalculates IDF values
        try:
            with self.dal_provider.connection() as conn:
                # Get total number of documents
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM commands")
                total_docs = cursor.fetchone()[0]
                
                if total_docs == 0:
                    logger.warning("No documents found for IDF calculation")
                    context.idf_values = {}
                    return
                
                # Calculate document frequencies for all terms
                cursor.execute("""
                    SELECT term, COUNT(DISTINCT command_id) as doc_freq
                    FROM term_frequencies 
                    GROUP BY term
                """)
                
                idf_values = {}
                for term, doc_freq in cursor.fetchall():
                    # Calculate IDF: log(N/df) where N is total docs, df is doc frequency
                    idf = np.log(total_docs / doc_freq) if doc_freq > 0 else 0.0
                    idf_values[term] = idf
                
                context.idf_values = idf_values
                
                # Update corpus statistics
                self._update_corpus_stats(conn, total_docs)
                
                logger.debug(f"Calculated IDF values for {len(idf_values)} terms")
                
        except Exception as e:
            logger.error(f"Error calculating IDF values: {e}")
            context.idf_values = {}
        
        self.execution_time = time.time() - start_time
        context.processing_times[self.name] = self.execution_time
        
        logger.debug(f"Calculated IDF values in {self.execution_time:.2f}s")
    
    def _update_corpus_stats(self, conn, total_docs):
        """Update corpus statistics in the database."""
        cursor = conn.cursor()
        
        # Calculate average document length
        cursor.execute("SELECT AVG(length) FROM commands WHERE length > 0")
        avg_length_result = cursor.fetchone()
        avg_length = avg_length_result[0] if avg_length_result and avg_length_result[0] else 1.0
        
        # Update corpus stats table
        cursor.execute("""
            INSERT OR REPLACE INTO corpus_stats (total_docs, avg_doc_length)
            VALUES (?, ?)
        """, (total_docs, avg_length))
        
        logger.debug(f"Updated corpus stats: {total_docs} docs, avg length {avg_length:.2f}")
    
    def _default_tokenize(self, text):
        """Default tokenization - simple word splitting."""
        import re
        return re.findall(r'\b\w+\b', text.lower())


class BuildANNIndexCommand(ProcessingCommand):
    """
    Command to build Approximate Nearest Neighbor (ANN) index.
    
    Creates a searchable K-means clustering index from embeddings for fast semantic search.
    """
    
    def __init__(self, dal_provider, ann_index_handler, min_commands_for_clustering: int = 50, 
                 n_clusters: int = 20, embedding_dtype=np.float32):
        super().__init__("BuildANNIndex") 
        self.dal_provider = dal_provider
        self.ann_index_handler = ann_index_handler
        self.min_commands_for_clustering = min_commands_for_clustering
        self.n_clusters = n_clusters
        self.embedding_dtype = embedding_dtype
        
    def execute(self, context: ProcessingContext) -> None:
        """Build ANN index from all embeddings in database."""
        start_time = time.time()
        
        logger.debug("Building ANN index from database embeddings")
        
        try:
            # Load all embeddings from database
            all_data = self.dal_provider.command_dal.get_all_commands_with_embeddings_for_clustering()
            
            if len(all_data) < self.min_commands_for_clustering:
                logger.info(f"Too few commands ({len(all_data)}) for ANN clustering, skipping")
                context.ann_index_data = None
                self.execution_time = time.time() - start_time
                context.processing_times[self.name] = self.execution_time
                return
            
            # Progress callback for loading phase
            if context.progress_callback:
                context.progress_callback(67, "Loading embeddings...")
            
            # Extract embeddings from database format
            embeddings_list = []
            for i, (_, _, emb) in enumerate(all_data):
                # Dequantize embedding based on storage format
                if self.embedding_dtype == np.int8:
                    stored_emb = np.frombuffer(emb, dtype=np.int8)[:384]
                    # Scale back from int8 range to float32
                    dequantized = stored_emb.astype(np.float32) / 127.0
                elif self.embedding_dtype == np.float16:
                    stored_emb = np.frombuffer(emb, dtype=np.float16)[:384]
                    # Convert FP16 to FP32 for computation
                    dequantized = stored_emb.astype(np.float32)
                elif self.embedding_dtype == np.float32:
                    stored_emb = np.frombuffer(emb, dtype=np.float32)[:384]
                    # Already FP32, use directly
                    dequantized = stored_emb.astype(np.float32)
                else:
                    raise ValueError(f"Unsupported embedding dtype: {self.embedding_dtype}")
                
                embeddings_list.append(dequantized)
                
                # Progress update during embedding extraction
                if context.progress_callback and i % max(1, len(all_data) // 10) == 0:
                    progress = 75 + int((i / len(all_data)) * 10)  # 75-85% range
                    context.progress_callback(progress, f"Processing embeddings... ({i+1}/{len(all_data)})")
            
            # Progress callback for indexing phase
            if context.progress_callback:
                context.progress_callback(85, "Building search index...")
            
            # Stack embeddings into numpy array
            embeddings_array = np.vstack(embeddings_list)
            
            # Train the ANN index (K-means clustering)
            logger.info(f"Training ANN index with {len(embeddings_array)} embeddings ({self.n_clusters} clusters)")
            self.ann_index_handler.fit(embeddings_array)
            
            # Store index data in context for potential saving
            context.ann_index_data = {
                "index_built": True,
                "num_embeddings": len(embeddings_array),
                "n_clusters": self.n_clusters,
                "is_trained": self.ann_index_handler.is_trained
            }
            
            logger.info(f"ANN index built successfully with {len(embeddings_array)} embeddings")
            
        except Exception as e:
            logger.error(f"Error building ANN index: {e}")
            context.ann_index_data = None
            raise
        
        self.execution_time = time.time() - start_time
        context.processing_times[self.name] = self.execution_time
        
        logger.debug(f"Built ANN index in {self.execution_time:.2f}s")
    
    def can_execute(self, context: ProcessingContext) -> bool:
        # This command works with database embeddings, not context embeddings
        return True


class ClearCacheCommand(ProcessingCommand):
    """
    Command to clear various caches (ANN index, cluster cache, etc.).
    
    Used when rebuilding indexes or when data has changed.
    """
    
    def __init__(self, db_path: str):
        super().__init__("ClearCache")
        self.db_path = db_path
        
    def execute(self, context: ProcessingContext) -> None:
        """Clear ANN and cluster caches."""
        start_time = time.time()
        
        caches_cleared = []
        
        # Clear cluster cache
        cluster_cache_path = self.db_path.replace('.db', '_clusters.pkl')
        if os.path.exists(cluster_cache_path):
            os.remove(cluster_cache_path)
            caches_cleared.append("cluster")
            logger.debug("Cleared cluster cache")
            
        # Clear ANN index cache  
        ann_cache_path = self.db_path.replace('.db', '_ann_index.pkl')
        if os.path.exists(ann_cache_path):
            os.remove(ann_cache_path)
            caches_cleared.append("ann_index")
            logger.debug("Cleared ANN index cache")
        
        self.execution_time = time.time() - start_time
        context.processing_times[self.name] = self.execution_time
        
        if caches_cleared:
            logger.info(f"Cleared caches: {', '.join(caches_cleared)}")
        else:
            logger.debug("No caches found to clear")


class DatabaseWriteCommand(ProcessingCommand):
    """
    Command to write processed data to the database using bulk operations.
    
    Uses the DAL bulk_operations context manager for optimal performance.
    Writes commands, embeddings, and term frequencies in bulk.
    """
    
    def __init__(self, dal_provider, tokenizer_func=None):
        super().__init__("DatabaseWrite")
        self.dal_provider = dal_provider
        self.tokenizer_func = tokenizer_func or self._default_tokenize
        
    def execute(self, context: ProcessingContext) -> None:
        """Write processed data to database using bulk operations."""
        start_time = time.time()
        
        if not context.commands or not context.processed_embeddings:
            raise ValueError("Missing commands or processed embeddings for database write")
            
        if len(context.commands) != len(context.processed_embeddings):
            raise ValueError("Mismatch between commands and embeddings count")
        
        logger.debug(f"Writing {len(context.commands)} commands with embeddings to database")
        
        # Progress callback for database operations
        if context.progress_callback:
            context.progress_callback(80, "Saving to database...")
        
        try:
            # Step 1: Insert commands and get their IDs directly
            command_data = [(cmd, len(cmd.split())) for cmd in context.commands]
            command_ids = self.dal_provider.command_dal.add_commands_batch(command_data)
            
            if len(command_ids) != len(context.commands):
                raise ValueError(f"Command insertion mismatch: expected {len(context.commands)} IDs, got {len(command_ids)}")
            
            # Step 2: Prepare embedding data using returned IDs
            embedding_data = []
            for i, command_id in enumerate(command_ids):
                if i < len(context.processed_embeddings):
                    # Convert quantized embedding to bytes for storage
                    quantized_embedding = context.processed_embeddings[i]
                    embedding_data.append((command_id, quantized_embedding.tobytes()))
                else:
                    logger.warning(f"Missing embedding for command at index {i}")
            
            # Progress callback for embedding write
            if context.progress_callback:
                context.progress_callback(85, "Saving embeddings...")
            
            # Step 3: Bulk insert embeddings
            self.dal_provider.command_dal.add_embeddings_batch(embedding_data)
            
            # Step 4: Prepare term frequency data for BM25
            term_freq_data = []
            for i, command in enumerate(context.commands):
                if i < len(command_ids):
                    command_id = command_ids[i]
                    # Tokenize command and count term frequencies
                    terms = self.tokenizer_func(command)
                    from collections import Counter
                    term_counts = Counter(terms)
                    
                    for term, freq in term_counts.items():
                        term_freq_data.append((term, command_id, freq))
                else:
                    logger.warning(f"Missing command ID for command at index {i}")
            
            # Progress callback for term frequency write
            if context.progress_callback:
                context.progress_callback(90, "Saving keyword index...")
            
            # Step 5: Bulk insert term frequencies
            self.dal_provider.command_dal.add_term_frequencies_batch(term_freq_data)
            
            # Store results in context
            context.command_ids = command_ids
            
            logger.debug(f"Successfully wrote {len(context.commands)} commands, "
                        f"{len(embedding_data)} embeddings, and {len(term_freq_data)} term frequencies")
            
        except Exception as e:
            logger.error(f"Error writing data to database: {e}")
            raise
        
        self.execution_time = time.time() - start_time
        context.processing_times[self.name] = self.execution_time
        
        logger.debug(f"Wrote data to database in {self.execution_time:.2f}s")
    
    def _default_tokenize(self, text):
        """Default tokenization - simple word splitting."""
        import re
        return re.findall(r'\b\w+\b', text.lower())
    
    def can_execute(self, context: ProcessingContext) -> bool:
        return (context.commands is not None and 
                context.processed_embeddings is not None and
                len(context.commands) == len(context.processed_embeddings))