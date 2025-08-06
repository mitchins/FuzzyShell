"""
Onboarding flow with clean view/viewmodel separation.
The view delegates to existing download/embedding logic.
"""

import os
import time
import threading
import urwid
from typing import Callable, Optional
from .screens.onboarding import OnboardingScreen, StartupManager, OnboardingStage, ONBOARDING_PALETTE


class OnboardingViewModel:
    """ViewModel that delegates to existing FuzzyShell download/embedding logic."""
    
    def __init__(self, fuzzyshell_instance):
        self.fuzzyshell = fuzzyshell_instance
        self.startup_manager = StartupManager()
        self.is_running = False
        self.error = None
        
    def needs_onboarding(self) -> bool:
        """Check if onboarding is needed by delegating to file system check."""
        model_dir = os.path.expanduser("~/.fuzzyshell/model/")
        desc_model_dir = os.path.expanduser("~/.fuzzyshell/description_model/")
        
        model_exists = os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0
        desc_exists = os.path.exists(desc_model_dir) and len(os.listdir(desc_model_dir)) > 0
        
        return not (model_exists and desc_exists)
    
    def start_setup(self, completion_callback: Callable):
        """Start the setup process by delegating to existing FuzzyShell logic."""
        if self.is_running:
            return
            
        self.is_running = True
        self.error = None
        
        # Run setup in background thread
        setup_thread = threading.Thread(
            target=self._run_setup_async, 
            args=(completion_callback,), 
            daemon=True
        )
        setup_thread.start()
    
    def _run_setup_async(self, completion_callback: Callable):
        """Run the actual setup by delegating to existing methods."""
        try:
            # Stage 1: Check models
            self.startup_manager.update_progress(
                OnboardingStage.CHECKING_MODELS, 5, 
                "Checking for models..."
            )
            time.sleep(0.5)
            
            # Stage 2: Delegate to existing FuzzyShell model initialization
            # This is where the existing download logic runs
            self.startup_manager.update_progress(
                OnboardingStage.DOWNLOADING_EMBEDDING, 25,
                "Initializing models (this may download files)..."
            )
            
            # DELEGATE: This calls your existing download/model logic
            model_ready = self._delegate_to_existing_setup()
            
            if not model_ready:
                raise Exception("Model setup failed")
            
            # Stage 3: Complete
            self.startup_manager.update_progress(
                OnboardingStage.COMPLETE, 100, 
                "Setup complete!"
            )
            time.sleep(1.5)
            
            # Notify completion
            completion_callback(True, None)
            
        except Exception as e:
            self.error = str(e)
            self.startup_manager.set_error(self.error)
            time.sleep(2)
            completion_callback(False, self.error)
        finally:
            self.is_running = False
    
    def _delegate_to_existing_setup(self) -> bool:
        """
        DELEGATE: Call the existing FuzzyShell setup logic.
        This is where all your existing download code runs unchanged.
        """
        try:
            # This calls your existing model initialization which handles:
            # - Checking if models exist
            # - Downloading if needed  
            # - Setting up embeddings
            # - All the existing logic in ModelHandler/DescriptionHandler
            return self.fuzzyshell.wait_for_model(timeout=60.0)
        except Exception:
            return False


class OnboardingFlow:
    """The view that orchestrates onboarding with clean delegation."""
    
    def __init__(self, fuzzyshell_instance):
        self.viewmodel = OnboardingViewModel(fuzzyshell_instance)
        self.loop = None
        self.screen = None
        self.result = None
        
    def run_if_needed(self) -> bool:
        """
        Run onboarding if needed, otherwise just return True.
        
        Returns:
            True if ready to proceed (no onboarding needed or completed successfully)
            False if onboarding failed or was cancelled
        """
        if not self.viewmodel.needs_onboarding():
            return True
            
        return self._run_onboarding_tui()
    
    def _run_onboarding_tui(self) -> bool:
        """Run the onboarding TUI."""
        # Create view (screen)
        self.screen = OnboardingScreen(self.viewmodel.startup_manager)
        
        # Create TUI loop
        self.loop = urwid.MainLoop(
            self.screen,
            palette=ONBOARDING_PALETTE,
            unhandled_input=self._handle_input
        )
        
        # Start the setup process (viewmodel delegates to existing logic)
        self.viewmodel.start_setup(self._on_setup_complete)
        
        # Run TUI
        try:
            self.loop.run()
            return self.result is True
        except Exception:
            return False
    
    def _handle_input(self, key):
        """Handle user input during onboarding."""
        if key == 'esc':
            self.result = False
            self._exit_tui()
    
    def _on_setup_complete(self, success: bool, error: Optional[str]):
        """Called when setup completes (from viewmodel)."""
        self.result = success
        if self.loop:
            # Schedule TUI exit on next loop iteration
            self.loop.set_alarm_in(0.1, lambda loop, data: self._exit_tui())
    
    def _exit_tui(self):
        """Exit the TUI."""
        if self.loop:
            raise urwid.ExitMainLoop()


# Simple integration function
def run_fuzzyshell_with_onboarding(fuzzyshell_instance, main_callback: Callable):
    """
    Run FuzzyShell with onboarding flow if needed.
    
    Args:
        fuzzyshell_instance: FuzzyShell instance to set up
        main_callback: Function to call after setup (should run main TUI)
    
    Returns:
        Result from main_callback, or False if setup failed
    """
    flow = OnboardingFlow(fuzzyshell_instance)
    
    if flow.run_if_needed():
        # Setup complete or not needed, run main app
        return main_callback()
    else:
        # Setup failed or cancelled
        return False