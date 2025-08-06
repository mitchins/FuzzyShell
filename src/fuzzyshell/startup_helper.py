"""
Startup helper for FuzzyShell with onboarding integration.
Handles first-time setup and model downloads with TUI progress.
"""

import os
import time
import urwid
import threading
from typing import Optional

from .tui.screens.onboarding import (
    OnboardingScreen, StartupManager, OnboardingStage, 
    ONBOARDING_PALETTE, create_model_download_progress_callback
)


class FuzzyShellStartup:
    """Manages FuzzyShell startup with onboarding TUI when needed."""
    
    def __init__(self, fuzzyshell_instance=None):
        self.fuzzyshell = fuzzyshell_instance
        self.startup_manager = StartupManager()
        self.onboarding_screen = None
        self.loop = None
        self.startup_complete = False
        self.startup_error = None
        
    def needs_onboarding(self) -> bool:
        """Check if onboarding is needed (models not downloaded)."""
        if not self.fuzzyshell:
            return False
            
        try:
            # Check if models exist
            model_dir = os.path.expanduser("~/.fuzzyshell/model/")
            desc_model_dir = os.path.expanduser("~/.fuzzyshell/description_model/")
            
            has_embedding_model = os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0
            has_desc_model = os.path.exists(desc_model_dir) and len(os.listdir(desc_model_dir)) > 0
            
            return not (has_embedding_model and has_desc_model)
        except Exception:
            return True  # If we can't check, assume onboarding is needed
    
    def run_onboarding_tui(self) -> bool:
        """Run the onboarding TUI. Returns True if successful, False if cancelled."""
        if not self.needs_onboarding():
            return True
            
        self.startup_manager.update_progress(
            OnboardingStage.CHECKING_MODELS, 0, 
            "Checking for required models..."
        )
        
        # Create onboarding screen
        self.onboarding_screen = OnboardingScreen(self.startup_manager)
        
        # Create TUI loop
        palette = ONBOARDING_PALETTE
        self.loop = urwid.MainLoop(
            self.onboarding_screen,
            palette=palette,
            unhandled_input=self._handle_onboarding_input
        )
        
        # Start model download in background thread
        download_thread = threading.Thread(target=self._download_models_async, daemon=True)
        download_thread.start()
        
        try:
            # Run the TUI
            self.loop.run()
            return self.startup_complete and not self.startup_error
        except Exception as e:
            self.startup_error = str(e)
            return False
    
    def _handle_onboarding_input(self, key):
        """Handle input during onboarding."""
        if key == 'esc' or key == 'cancel_onboarding':
            self.startup_error = "Cancelled by user"
            raise urwid.ExitMainLoop()
    
    def _download_models_async(self):
        """Download models asynchronously with progress updates."""
        try:
            # Stage 1: Download embedding model
            self.startup_manager.update_progress(
                OnboardingStage.DOWNLOADING_EMBEDDING, 10,
                "Downloading embedding model..."
            )
            
            # Create progress callback for model handler
            progress_callback = create_model_download_progress_callback(self.startup_manager)
            
            # Simulate or actually call model initialization
            if self.fuzzyshell:
                try:
                    # This should trigger model downloads if needed
                    # We'll need to modify ModelHandler to accept progress callbacks
                    self._init_models_with_progress(progress_callback)
                except Exception as e:
                    self.startup_manager.set_error(f"Model download failed: {str(e)}")
                    return
            else:
                # Fallback simulation for testing
                self._simulate_model_download()
            
            # Stage 3: Finalize
            self.startup_manager.update_progress(
                OnboardingStage.COMPLETE, 100,
                "FuzzyShell is ready!"
            )
            
            # Give user a moment to see completion
            time.sleep(2)
            
            self.startup_complete = True
            if self.loop:
                self.loop.set_alarm_in(0.1, lambda loop, data: self._exit_onboarding())
                
        except Exception as e:
            self.startup_manager.set_error(str(e))
            self.startup_error = str(e)
    
    def _init_models_with_progress(self, progress_callback):
        """Initialize models with progress tracking."""
        # Stage 1: Embedding model (50% of progress)
        progress_callback('embedding', 1, 2, 'Embedding Model')
        
        # Initialize embedding model (this may trigger download)
        if hasattr(self.fuzzyshell, '_model') and self.fuzzyshell._model is None:
            # Force model initialization
            self.fuzzyshell.wait_for_model(timeout=60.0)
        
        time.sleep(1)  # Brief pause for UI
        
        # Stage 2: Description model (remaining 50%)
        progress_callback('description', 2, 2, 'Description Model')
        
        # Initialize description model (this may trigger download)
        if hasattr(self.fuzzyshell, '_description_handler'):
            try:
                from .model_handler import DescriptionHandler
                desc_handler = DescriptionHandler()
                # This may trigger download
                time.sleep(2)  # Allow download time
            except Exception:
                pass
        
        # Stage 3: Initialization complete
        progress_callback('initializing', 2, 2, 'Ready')
    
    def _simulate_model_download(self):
        """Simulate model download for testing/fallback."""
        stages = [
            (OnboardingStage.DOWNLOADING_EMBEDDING, 25, "Downloading embedding model..."),
            (OnboardingStage.DOWNLOADING_EMBEDDING, 50, "Installing embedding model..."),
            (OnboardingStage.DOWNLOADING_DESCRIPTION, 75, "Downloading description model..."),
            (OnboardingStage.INITIALIZING, 90, "Initializing models..."),
        ]
        
        for stage, progress, message in stages:
            self.startup_manager.update_progress(stage, progress, message)
            time.sleep(1.5)  # Simulate work
    
    def _exit_onboarding(self):
        """Exit the onboarding TUI."""
        if self.loop:
            raise urwid.ExitMainLoop()


def run_startup_with_onboarding(fuzzyshell_instance) -> bool:
    """
    Run FuzzyShell startup with onboarding if needed.
    
    Returns:
        True if startup successful (or no onboarding needed)
        False if startup failed or was cancelled
    """
    startup = FuzzyShellStartup(fuzzyshell_instance)
    
    if not startup.needs_onboarding():
        return True
    
    return startup.run_onboarding_tui()