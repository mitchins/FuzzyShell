"""
Progress wrapper for FuzzyShell model downloads.
Provides a way to monitor download progress without major code changes.
"""

import logging
import threading
import time
from typing import Optional, Callable


class ProgressMonitor:
    """Monitors log output to track download progress."""
    
    def __init__(self, progress_callback: Optional[Callable] = None):
        self.progress_callback = progress_callback
        self.current_stage = "checking"
        self.total_models = 2  # embedding + description
        self.current_model = 0
        self.model_names = ["Embedding Model", "Description Model"]
        self.monitoring = False
        self.original_log_level = None
        
    def start_monitoring(self):
        """Start monitoring log output for download progress."""
        if not self.progress_callback:
            return
            
        self.monitoring = True
        
        # Set up log interception
        logger = logging.getLogger('FuzzyShell.ModelHandler')
        self.original_log_level = logger.level
        
        # Add our custom handler
        self.log_handler = ProgressLogHandler(self)
        logger.addHandler(self.log_handler)
        
        # Update initial stage
        self._update_progress("checking", 0, "Checking for models...")
    
    def stop_monitoring(self):
        """Stop monitoring and clean up."""
        self.monitoring = False
        
        # Remove our custom handler
        if hasattr(self, 'log_handler'):
            logger = logging.getLogger('FuzzyShell.ModelHandler')
            logger.removeHandler(self.log_handler)
    
    def _update_progress(self, stage: str, percent: int, message: str = ""):
        """Update progress via callback."""
        if self.progress_callback and self.monitoring:
            try:
                self.progress_callback(stage, self.current_model, self.total_models, percent, message)
            except Exception:
                pass  # Don't let callback errors break monitoring


class ProgressLogHandler(logging.Handler):
    """Custom log handler to intercept model download messages."""
    
    def __init__(self, monitor: ProgressMonitor):
        super().__init__()
        self.monitor = monitor
        self.last_download_percent = 0
        
    def emit(self, record):
        """Process log records for download progress."""
        if not self.monitor.monitoring:
            return
            
        message = record.getMessage()
        
        try:
            if "Downloading custom terminal-trained model" in message:
                self.monitor.current_model = 1
                self.monitor.current_stage = "downloading_embedding"
                self.monitor._update_progress("downloading_embedding", 0, "Downloading embedding model...")
                
            elif "Custom terminal-trained model download complete" in message:
                self.monitor._update_progress("downloading_embedding", 50, "Embedding model downloaded")
                
            elif "Downloading custom terminal-trained tokenizer" in message:
                self.monitor._update_progress("downloading_embedding", 60, "Downloading tokenizer...")
                
            elif "tokenizer.json download complete" in message:
                self.monitor._update_progress("downloading_embedding", 70, "Tokenizer downloaded")
                
            elif "Model files checked" in message:
                self.monitor._update_progress("downloading_embedding", 100, "Embedding model ready")
                
            elif "ModelHandler initialization complete" in message:
                self.monitor.current_model = 2
                self.monitor.current_stage = "downloading_description" 
                self.monitor._update_progress("downloading_description", 100, "Models initialized")
                
            elif "Downloaded" in message and "MB" in message:
                # Parse download progress from "Downloaded X.XX/YY.YY MB"
                try:
                    parts = message.split()
                    if len(parts) >= 2:
                        downloaded_part = parts[1]  # "X.XX/YY.YY"
                        if '/' in downloaded_part:
                            current, total = downloaded_part.split('/')
                            current_mb = float(current)
                            total_mb = float(total)
                            percent = min(95, int((current_mb / total_mb) * 50))  # Cap at 50% for embedding model
                            
                            # Only update every 5% to avoid spam
                            if percent >= self.last_download_percent + 5:
                                self.last_download_percent = percent
                                self.monitor._update_progress("downloading_embedding", percent, 
                                                           f"Downloaded {current_mb:.1f}/{total_mb:.1f} MB")
                except (ValueError, IndexError):
                    pass
                    
        except Exception:
            pass  # Don't let parsing errors break the flow


def create_progress_monitored_startup(fuzzyshell_instance, progress_callback):
    """
    Create a startup flow that monitors download progress.
    
    Args:
        fuzzyshell_instance: The FuzzyShell instance to initialize
        progress_callback: Function called with (stage, current_model, total_models, percent, message)
    
    Returns:
        Function that runs the monitored startup
    """
    
    def run_monitored_startup():
        monitor = ProgressMonitor(progress_callback)
        
        try:
            # Start monitoring
            monitor.start_monitoring()
            
            # Check if models need downloading
            needs_download = False
            try:
                # Try to initialize model - this will trigger downloads if needed
                model_ready = fuzzyshell_instance.wait_for_model(timeout=60.0)
                if not model_ready:
                    raise Exception("Model initialization failed")
                    
                # If we get here, models are ready
                monitor._update_progress("complete", 2, 2, 100, "FuzzyShell ready!")
                
            except Exception as e:
                monitor._update_progress("error", 0, 2, 0, f"Setup failed: {str(e)}")
                raise
                
        finally:
            monitor.stop_monitoring()
    
    return run_monitored_startup


def run_startup_with_progress_tui(fuzzyshell_instance):
    """
    Run FuzzyShell startup with progress TUI if models need downloading.
    
    Returns:
        True if successful, False if failed or cancelled
    """
    from .startup_helper import FuzzyShellStartup
    
    # Check if onboarding is needed
    startup = FuzzyShellStartup(fuzzyshell_instance)
    if not startup.needs_onboarding():
        # Models already exist, just initialize normally
        try:
            fuzzyshell_instance.wait_for_model(timeout=10.0)
            return True
        except:
            return False
    
    # Models need downloading - show progress TUI
    def progress_callback(stage, current_model, total_models, percent, message):
        # Map our progress to the startup manager
        stage_map = {
            'checking': startup.startup_manager.OnboardingStage.CHECKING_MODELS,
            'downloading_embedding': startup.startup_manager.OnboardingStage.DOWNLOADING_EMBEDDING, 
            'downloading_description': startup.startup_manager.OnboardingStage.DOWNLOADING_DESCRIPTION,
            'complete': startup.startup_manager.OnboardingStage.COMPLETE,
            'error': startup.startup_manager.OnboardingStage.CHECKING_MODELS
        }
        
        onboarding_stage = stage_map.get(stage, startup.startup_manager.OnboardingStage.CHECKING_MODELS)
        startup.startup_manager.update_progress(onboarding_stage, percent, message)
    
    # Create monitored startup
    monitored_startup = create_progress_monitored_startup(fuzzyshell_instance, progress_callback)
    
    # Replace the startup's download method with our monitored version
    startup._download_models_async = lambda: monitored_startup()
    
    return startup.run_onboarding_tui()