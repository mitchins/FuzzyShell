"""
Simple onboarding integration that captures all downloads.
Ensures TUI shows for any first-time setup, no CLI fallback.
"""

import os
import logging
import threading
import time
import urwid
from typing import Callable, Optional
from .screens.onboarding import OnboardingScreen, StartupManager, OnboardingStage, ONBOARDING_PALETTE


class DownloadCapture:
    """Captures download progress from logs and converts to TUI updates."""
    
    def __init__(self, startup_manager: StartupManager):
        self.startup_manager = startup_manager
        self.log_handler = None
        self.last_percent = 0
        self.current_stage = OnboardingStage.CHECKING_MODELS
        self.stage_weights = {
            OnboardingStage.CHECKING_MODELS: (0, 5),      # 0-5%
            OnboardingStage.DOWNLOADING_EMBEDDING: (5, 70),   # 5-70% 
            OnboardingStage.DOWNLOADING_DESCRIPTION: (70, 95), # 70-95%
            OnboardingStage.INITIALIZING: (95, 100),     # 95-100%
        }
        
    def start_capture(self):
        """Start capturing download logs."""
        # Add log handler to capture ModelHandler output
        logger = logging.getLogger('FuzzyShell.ModelHandler')
        self.log_handler = LogProgressHandler(self)
        logger.addHandler(self.log_handler)
        logger.setLevel(logging.DEBUG)  # Ensure we capture debug logs
        
    def stop_capture(self):
        """Stop capturing logs."""
        if self.log_handler:
            logger = logging.getLogger('FuzzyShell.ModelHandler')
            logger.removeHandler(self.log_handler)
            
    def _map_to_overall_progress(self, stage: OnboardingStage, stage_percent: int) -> int:
        """Map stage-specific progress to overall 0-100% progress."""
        start, end = self.stage_weights.get(stage, (0, 100))
        stage_range = end - start
        return start + int((stage_percent / 100) * stage_range)
    
    def update_from_log(self, message: str):
        """Update progress based on log message with aggregated progress."""
        try:
            if "Downloading custom terminal-trained model" in message:
                self.current_stage = OnboardingStage.DOWNLOADING_EMBEDDING
                overall_progress = self._map_to_overall_progress(self.current_stage, 0)
                self.startup_manager.update_progress(self.current_stage, overall_progress, "Starting model download...")
                
            elif "Downloaded" in message and "MB" in message:
                # Parse download progress and map to overall progress
                try:
                    parts = message.split()
                    if len(parts) >= 2:
                        downloaded_part = parts[1]  # "X.XX/YY.YY"
                        if '/' in downloaded_part:
                            current, total = downloaded_part.split('/')
                            current_mb = float(current)
                            total_mb = float(total)
                            
                            # Stage progress: 0-90% for download
                            stage_percent = min(90, int((current_mb / total_mb) * 90))
                            overall_progress = self._map_to_overall_progress(self.current_stage, stage_percent)
                            
                            if overall_progress > self.last_percent:
                                self.last_percent = overall_progress
                                self.startup_manager.update_progress(
                                    self.current_stage, overall_progress,
                                    f"Downloading: {current_mb:.1f}/{total_mb:.1f} MB"
                                )
                except (ValueError, IndexError):
                    pass
                    
            elif "model download complete" in message:
                overall_progress = self._map_to_overall_progress(self.current_stage, 100)
                self.startup_manager.update_progress(self.current_stage, overall_progress, "Embedding model ready")
                
            elif "tokenizer.json download complete" in message:
                self.current_stage = OnboardingStage.DOWNLOADING_DESCRIPTION
                overall_progress = self._map_to_overall_progress(self.current_stage, 25)
                self.startup_manager.update_progress(self.current_stage, overall_progress, "Setting up tokenizer...")
                
            elif "ModelHandler initialization complete" in message:
                self.current_stage = OnboardingStage.INITIALIZING
                overall_progress = self._map_to_overall_progress(self.current_stage, 100)
                self.startup_manager.update_progress(self.current_stage, overall_progress, "Initialization complete!")
                
        except Exception:
            pass  # Don't let log parsing break the flow


class LogProgressHandler(logging.Handler):
    """Custom log handler to capture download progress."""
    
    def __init__(self, capture: DownloadCapture):
        super().__init__()
        self.capture = capture
        
    def emit(self, record):
        """Process log records."""
        try:
            self.capture.update_from_log(record.getMessage())
        except Exception:
            pass


def check_needs_setup() -> bool:
    """Check if any setup is needed - models OR ingestion."""
    # Check models - use active model configuration
    try:
        from fuzzyshell.model_configs import get_required_files, get_model_config
        
        model_dir = os.path.expanduser("~/.fuzzyshell/model/")
        config = get_model_config()  # Get active model config
        
        # Check the main model file path (could be nested like onnx/int8/model_quantized.onnx)
        model_path = os.path.join(model_dir, config["files"]["model_path"])
        model_exists = os.path.exists(model_path)
        
        # Check tokenizer files (these are in the root model directory)
        required_files = get_required_files()  # Uses active model
        tokenizer_files_exist = all(
            os.path.exists(os.path.join(model_dir, filename))
            for filename in required_files
            if filename != os.path.basename(config["files"]["model_path"])  # Skip model file, we checked it above
        )
        
        models_exist = model_exists and tokenizer_files_exist
    except Exception:
        # If we can't check model config, assume models are missing
        models_exist = False
    
    # Check database/ingestion status
    db_path = os.path.expanduser("~/.fuzzyshell/fuzzyshell.db")
    db_exists = os.path.exists(db_path)
    
    # If either models are missing OR database doesn't exist, show onboarding
    return not (models_exist and db_exists)


def run_with_onboarding(fuzzyshell_instance, main_tui_callback: Callable) -> bool:
    """
    Run with onboarding TUI that captures ALL download activity.
    
    Args:
        fuzzyshell_instance: FuzzyShell instance (should NOT be initialized yet)
        main_tui_callback: Function to call after setup is complete
        
    Returns:
        True if successful, False if failed or cancelled
    """
    
    if not check_needs_setup():
        # No setup needed, run main TUI directly
        return main_tui_callback()
    
    # Setup needed - show onboarding TUI
    startup_manager = StartupManager()
    download_capture = DownloadCapture(startup_manager)
    onboarding_screen = OnboardingScreen(startup_manager)
    
    setup_complete = False
    setup_error = None
    
    def handle_input(key):
        nonlocal setup_error
        if key == 'esc':
            setup_error = "Setup cancelled by user"
            raise urwid.ExitMainLoop()
    
    # Create TUI loop
    loop = urwid.MainLoop(
        onboarding_screen,
        palette=ONBOARDING_PALETTE,
        unhandled_input=handle_input
    )
    
    def run_setup_async():
        """Run setup in background, capturing all output."""
        nonlocal setup_complete, setup_error
        
        try:
            # Start capturing download logs
            download_capture.start_capture()
            
            # Initial progress
            startup_manager.update_progress(OnboardingStage.CHECKING_MODELS, 0, "Checking models...")
            time.sleep(0.5)
            
            # DELEGATE: Run existing setup logic (this triggers all downloads)
            # The DownloadCapture will translate log output to progress updates
            model_ready = fuzzyshell_instance.wait_for_model(timeout=120.0)
            
            if not model_ready:
                raise Exception("Model initialization failed")
            
            # Final stage
            startup_manager.update_progress(OnboardingStage.COMPLETE, 100, "Setup complete!")
            time.sleep(2)
            
            setup_complete = True
            
        except Exception as e:
            setup_error = f"Setup failed: {str(e)}"
            startup_manager.set_error(setup_error)
            time.sleep(2)
            
        finally:
            download_capture.stop_capture()
            # Exit TUI
            loop.set_alarm_in(0.1, lambda loop, data: loop.stop())
    
    # Start setup in background thread
    setup_thread = threading.Thread(target=run_setup_async, daemon=True)
    setup_thread.start()
    
    # Run onboarding TUI
    try:
        loop.run()
    except Exception:
        pass
    
    # Check results
    if setup_error:
        print(f"Setup failed: {setup_error}")
        return False
        
    if not setup_complete:
        print("Setup incomplete")
        return False
    
    # Setup successful, run main TUI
    return main_tui_callback()


# Usage example for integration:
def integrate_with_main():
    """
    Example integration into main FuzzyShell startup.
    Replace your current startup with this pattern.
    """
    from ..fuzzyshell import FuzzyShell
    from ..fuzzy_tui import run_ui
    
    def create_and_run_main_tui():
        # Create FuzzyShell instance (but don't initialize models yet)
        fuzzyshell = FuzzyShell()
        
        def run_main_tui():
            # Run the main TUI (models are already initialized by onboarding)
            return run_ui(lambda query: fuzzyshell.search(query), fuzzyshell)
        
        # This will show onboarding if needed, then run main TUI
        return run_with_onboarding(fuzzyshell, run_main_tui)
    
    return create_and_run_main_tui()