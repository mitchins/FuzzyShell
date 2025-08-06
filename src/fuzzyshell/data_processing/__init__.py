"""
Data Processing Pipeline using Command + Facade patterns.

This module contains the Gang of Four patterns for data processing operations:
- Command Pattern: Individual processing steps (ExtractEmbeddingsCommand, etc.)
- Facade Pattern: DataProcessingPipeline that hides complexity of orchestration
"""

from .pipeline import DataProcessingPipeline
from .commands import (
    ProcessingCommand,
    ExtractEmbeddingsCommand,
    CalculateIDFCommand, 
    BuildANNIndexCommand,
    ClearCacheCommand,
    DatabaseWriteCommand
)
from .context import ProcessingContext, ProcessingResult

__all__ = [
    'DataProcessingPipeline',
    'ProcessingCommand',
    'ExtractEmbeddingsCommand',
    'CalculateIDFCommand',
    'BuildANNIndexCommand', 
    'ClearCacheCommand',
    'DatabaseWriteCommand',
    'ProcessingContext',
    'ProcessingResult'
]