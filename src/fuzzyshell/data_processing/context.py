"""
Processing context and result classes for data processing pipeline.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np


@dataclass
class ProcessingContext:
    """
    Context object that carries state through the data processing pipeline.
    
    This is passed to each Command and accumulates results.
    """
    # Input data
    commands: List[str]
    embeddings: Optional[List[np.ndarray]] = None
    command_ids: Optional[List[int]] = None
    
    # Processing results
    processed_embeddings: Optional[List[np.ndarray]] = None
    idf_values: Optional[Dict[str, float]] = None
    term_frequencies: Optional[Dict[str, Dict[int, int]]] = None
    ann_index_data: Optional[Any] = None
    
    # Configuration
    batch_size: int = 1000
    use_clustering: bool = True
    progress_callback: Optional[callable] = None
    
    # Performance metrics
    processing_times: Dict[str, float] = None
    
    def __post_init__(self):
        if self.processing_times is None:
            self.processing_times = {}


@dataclass
class ProcessingResult:
    """
    Result of a data processing pipeline execution.
    """
    success: bool
    commands_processed: int
    embeddings_processed: int
    ann_index_built: bool
    processing_time_seconds: float
    error_message: Optional[str] = None
    
    # Detailed metrics
    embedding_extraction_time: float = 0.0
    idf_calculation_time: float = 0.0 
    ann_building_time: float = 0.0
    database_write_time: float = 0.0
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the processing result."""
        if not self.success:
            return f"Processing failed: {self.error_message}"
        
        return (f"Successfully processed {self.commands_processed} commands "
                f"and {self.embeddings_processed} embeddings in {self.processing_time_seconds:.2f}s. "
                f"ANN index: {'built' if self.ann_index_built else 'skipped'}")
    
    def get_performance_breakdown(self) -> Dict[str, float]:
        """Get detailed performance breakdown."""
        return {
            'embedding_extraction': self.embedding_extraction_time,
            'idf_calculation': self.idf_calculation_time,
            'ann_building': self.ann_building_time,
            'database_write': self.database_write_time,
            'total': self.processing_time_seconds
        }