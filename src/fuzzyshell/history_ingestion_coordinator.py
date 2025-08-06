"""
History Ingestion Coordinator for FuzzyShell.

Coordinates the complex process of ingesting shell history including:
- File reading and command cleaning
- TUI progress display management  
- Data pipeline integration
- Performance optimization
"""

import time
import threading
import logging
from typing import Optional, Callable, Set, List
from .fuzzy_tui import IngestionProgressTUI
from .error_handler_util import ErrorHandlerUtil

logger = logging.getLogger('FuzzyShell.HistoryIngestionCoordinator')


def tui_safe_print(*args, **kwargs):
    """Print function that works safely with TUI mode."""
    # Import here to avoid circular dependency
    from .fuzzyshell import _TUI_MODE_ACTIVE
    if not _TUI_MODE_ACTIVE:
        print(*args, **kwargs)


class HistoryIngestionCoordinator:
    """Coordinates the complex process of ingesting shell history."""
    
    def __init__(self, fuzzyshell_instance):
        """Initialize with reference to main FuzzyShell instance."""
        self.fuzzyshell = fuzzyshell_instance
        self.logger = logger
    
    def ingest_history(self, use_tui: bool = True, no_random: bool = False) -> Optional[int]:
        """
        Ingest shell history with optional TUI progress display.
        
        Args:
            use_tui: If True, use TUI progress display. If False or TUI fails, fall back to console.
            no_random: If True, disable random funny messages and show raw progress.
            
        Returns:
            Number of commands processed, or None if failed
        """
        start_time = time.time()
        history_file = self.fuzzyshell.get_shell_history_file()
        logger.info("Starting history ingestion from: %s", history_file)
        
        # Clear all caches when ingesting to ensure fresh results
        logger.info("Clearing caches due to data ingestion...")
        self.fuzzyshell.clear_all_caches()
        
        # Initialize model if needed and wait for it
        model_ready = self.fuzzyshell.wait_for_model(timeout=30.0)
        if not model_ready:
            logger.error("Model failed to load in time or initialization failed (timeout after 30s)")
            return None
        
        # Read and clean commands
        commands = self._read_and_clean_commands(history_file)
        if not commands:
            return None
        
        # Set up progress tracking
        tui_progress, console_fallback = self._setup_progress_tracking(use_tui, no_random, len(commands))
        
        # Optimize database for bulk operations
        self._optimize_database_for_bulk()
        
        try:
            # Process commands through data pipeline
            total_processed = self._process_commands_through_pipeline(
                commands, tui_progress, console_fallback
            )
            
            if total_processed is None:
                return None
            
            # Update metadata and finish
            self._finalize_ingestion(total_processed, tui_progress, console_fallback, start_time)
            
            return total_processed
            
        finally:
            # Always restore database settings
            self._restore_database_settings()
    
    def _read_and_clean_commands(self, history_file: str) -> Optional[List[str]]:
        """Read history file and clean commands."""
        try:
            with open(history_file, 'r', encoding='utf-8', errors='ignore') as f:
                commands = set(line.strip() for line in f)
        except FileNotFoundError:
            tui_safe_print(f"History file not found at: {history_file}")
            return None

        raw_command_count = len(commands)
        if raw_command_count == 0:
            tui_safe_print("No commands to process.")
            return None
        
        logger.info("Starting command processing...")
        
        # Filter and clean commands first, maintaining set for deduplication
        tui_safe_print(f"Cleaning {raw_command_count} raw commands...")
        clean_commands = set()
        for raw_command in commands:
            if not raw_command or raw_command.isspace():
                continue
            command = self.fuzzyshell.clean_shell_command(raw_command)
            if command and not command.isspace():
                clean_commands.add(command)
        
        if not clean_commands:
            tui_safe_print("No valid commands found after cleaning.")
            return None
        
        # Convert to list for batch processing (already deduplicated by set)
        final_commands = list(clean_commands)
        final_command_count = len(final_commands)
        
        tui_safe_print(f"After cleaning and deduplication: {final_command_count} unique commands to process")
        
        return final_commands
    
    def _setup_progress_tracking(self, use_tui: bool, no_random: bool, command_count: int):
        """Set up TUI or console progress tracking."""
        tui_progress = None
        console_fallback = False
        
        if use_tui:
            try:
                tui_progress = IngestionProgressTUI(no_random=no_random)
                tui_progress.set_total_commands(command_count)
                
                # Start TUI in a separate thread to avoid blocking
                def run_tui():
                    try:
                        loop = tui_progress.start()
                        loop.run()
                    except Exception as e:
                        logger.debug(f"TUI loop ended: {e}")
                
                tui_thread = threading.Thread(target=run_tui, daemon=True)
                tui_thread.start()
                
                # Give TUI a moment to initialize
                time.sleep(0.1)
                
                # Update initial status
                tui_progress.update_progress(0, "Starting batch processing...", "Initialization")
                
            except Exception as e:
                logger.warning(f"Failed to initialize TUI progress: {e}. Falling back to console.")
                console_fallback = True
                tui_progress = None
        else:
            console_fallback = True
        
        # If using console fallback, show simple message
        if console_fallback:
            tui_safe_print(f"Processing {command_count} commands...")
        
        return tui_progress, console_fallback
    
    def _optimize_database_for_bulk(self):
        """Enable SQLite fast mode for bulk ingestion."""
        try:
            self.fuzzyshell.command_dal.optimize_for_bulk_operations()
            logger.debug("Enabled SQLite fast mode for bulk ingestion")
        except Exception as e:
            logger.warning(f"Could not enable fast mode: {e}")
    
    def _process_commands_through_pipeline(self, commands: List[str], tui_progress, console_fallback) -> Optional[int]:
        """Process commands through the data pipeline."""
        if tui_progress:
            try:
                tui_progress.update_progress(10, f"Processing {len(commands)} unique commands", "Command Processing")
            except:
                pass
        
        if console_fallback:
            tui_safe_print("Processing commands with data pipeline...")
        
        # Create progress callback that works with both TUI and console
        def progress_callback(percent, message):
            # Map pipeline progress (0-100%) to ingestion progress (20-100%)
            adjusted_percent = 20 + int((percent / 100.0) * 80)
            
            if tui_progress:
                try:
                    tui_progress.update_progress(adjusted_percent, message, "Data Pipeline")
                except:
                    pass
            elif console_fallback:
                tui_safe_print(f"  {message} ({adjusted_percent}%)")
        
        # Run the complete data processing pipeline
        pipeline_result = self.fuzzyshell.data_pipeline.process_ingestion(
            commands=commands,
            progress_callback=progress_callback,
            use_clustering=True
        )
        
        if not pipeline_result.success:
            error_details = pipeline_result.error_message
            if console_fallback:
                tui_safe_print(f"❌ Data processing pipeline failed: {error_details}")
            ErrorHandlerUtil.log_and_raise_operation_error(
                operation_name="data processing pipeline",
                details=error_details,
                logger_instance=logger
            )
        
        total_processed = pipeline_result.commands_processed
        
        if tui_progress:
            tui_progress.processed_commands = total_processed
        
        logger.info(f"Pipeline processed {total_processed} commands in {pipeline_result.processing_time_seconds:.3f}s")
        logger.debug(f"Performance breakdown: {pipeline_result.get_performance_breakdown()}")
        
        return total_processed
    
    def _finalize_ingestion(self, total_processed: int, tui_progress, console_fallback, start_time: float):
        """Finalize ingestion with metadata updates and progress completion."""
        # Update metadata
        self.fuzzyshell._update_item_count()
        self.fuzzyshell.set_metadata('last_updated', time.strftime('%Y-%m-%d %H:%M:%S'))
        
        # Finish progress display
        if tui_progress:
            try:
                tui_progress.finish("Ingestion complete!")
            except:
                pass
        elif console_fallback:
            tui_safe_print(f"✅ Ingestion complete! {total_processed} commands processed.")
        
        # Log performance statistics
        total_time = time.time() - start_time
        logger.info("Ingestion complete: %d commands processed", total_processed)
        logger.info("Total ingestion time: %.3fs (%.3fs/command)", 
                   total_time, total_time/total_processed if total_processed > 0 else 0)
    
    def _restore_database_settings(self):
        """Restore original SQLite settings after bulk operations."""
        try:
            self.fuzzyshell.command_dal.restore_normal_pragmas()
            logger.debug("Restored SQLite settings after bulk ingestion")
        except Exception as e:
            logger.warning(f"Could not restore SQLite settings: {e}")