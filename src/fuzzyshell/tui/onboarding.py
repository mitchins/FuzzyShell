"""First-time setup onboarding with TUI progress display."""

import os
import time
import threading
import urwid
from typing import Callable

from .screens.onboarding import OnboardingScreen, StartupManager, OnboardingStage, ONBOARDING_PALETTE
from .breadcrumb_logger import screen_did_become_visible, screen_did_become_hidden
from .logging_redirect import start_tui_redirect, stop_tui_redirect


def run_comprehensive_onboarding(main_tui_callback: Callable, no_random=False):
    """Run first-time setup with TUI progress display."""
    
    # Enable TUI mode to suppress print statements
    from ..fuzzyshell import set_tui_mode
    set_tui_mode(True)
    
    # Start logging redirect
    start_tui_redirect()
    
    # Log screen transition
    screen_did_become_visible("OnboardingScreen", {"setup_type": "comprehensive"})
    
    # Create controller and screen
    startup_manager = StartupManager()
    onboarding_screen = OnboardingScreen(startup_manager)
    
    setup_complete = False
    setup_error = None
    
    def handle_input(key):
        nonlocal setup_error
        if key == 'esc':
            setup_error = "Setup cancelled by user"
            raise urwid.ExitMainLoop()
    
    def raise_exit_from_callback():
        raise urwid.ExitMainLoop()
    
    # Create TUI loop
    loop = urwid.MainLoop(
        onboarding_screen,
        palette=ONBOARDING_PALETTE,
        unhandled_input=handle_input
    )
    
    
    # Fun quips for entertainment vs technical details
    quips = [
        "Checking if we have our digital ducks in a row...",
        "Downloading more RAM... just kidding, it's models",
        "Teaching the computer to be fuzzy...",
        "Diving into your command history...",
        "Found some interesting commands in there...",
        "Polishing the final bits...",
        "Ready to fuzzy search like a boss!"
    ]
    
    technical_messages = [
        "Verifying model files and database...",
        "Loading embedding model (~50MB)...",
        "Model initialization complete",
        "Scanning shell history files...",
        "Indexed shell commands",
        "Optimizing search indices...",
        "Setup complete!"
    ]
    
    messages = technical_messages if no_random else quips
    
    steps = [
        {"stage": OnboardingStage.CHECKING_MODELS, "percent": 5, "message": messages[0]},
        {"stage": OnboardingStage.DOWNLOADING_EMBEDDING, "percent": 15, "message": messages[1]},
        {"stage": OnboardingStage.DOWNLOADING_EMBEDDING, "percent": 70, "message": messages[2]},
        {"stage": OnboardingStage.DOWNLOADING_DESCRIPTION, "percent": 75, "message": messages[3]},
        {"stage": OnboardingStage.DOWNLOADING_DESCRIPTION, "percent": 85, "message": messages[4]},
        {"stage": OnboardingStage.INITIALIZING, "percent": 95, "message": messages[5]},
        {"stage": OnboardingStage.COMPLETE, "percent": 100, "message": messages[6]},
    ]
    
    current_step_index = 0
    
    def ui_update(stage, percent, message):
        loop.set_alarm_in(0, lambda loop, user_data: startup_manager.update_progress(stage, percent, message))
    
    def advance_to_step(step_index, custom_message=None):
        """Advance to a specific step, optionally with custom message."""
        nonlocal current_step_index
        if step_index < len(steps):
            current_step_index = step_index
            step = steps[step_index]
            message = custom_message if custom_message else step["message"]
            ui_update(step["stage"], step["percent"], message)

    def setup_delegate():
        nonlocal setup_complete, setup_error
        
        try:
            advance_to_step(0)
            time.sleep(0.3)
            
            advance_to_step(1)
            from ..fuzzyshell import FuzzyShell
            fuzzyshell = FuzzyShell()
            
            model_ready = fuzzyshell.wait_for_model(timeout=120.0)
            if not model_ready:
                raise Exception("Model initialization failed")
            
            advance_to_step(2)
            advance_to_step(3)
            indexed_count = fuzzyshell.get_indexed_count()
            if indexed_count == 0:
                added_count = fuzzyshell.ingest_history(use_tui=False, no_random=True)
                if added_count and added_count > 0:
                    # Wait for database to be fully ready for searches
                    advance_to_step(4, "Finalizing search index...")
                    
                    # Ensure ANN index is properly saved and available
                    max_attempts = 10
                    for attempt in range(max_attempts):
                        try:
                            # Check if ANN manager is trained and cache is saved
                            if (fuzzyshell.ann_manager and 
                                fuzzyshell.ann_manager.is_trained() and
                                fuzzyshell.ann_manager.save_to_cache()):
                                break
                            time.sleep(0.3)
                        except:
                            time.sleep(0.3)
                            
                    # Verify ANN index won't trigger a rebuild on first search
                    try:
                        # Double-check ANN is fully ready and accessible
                        if (fuzzyshell.ann_manager and 
                            fuzzyshell.ann_manager.is_trained()):
                            # Perform a test search to verify no rebuild happens
                            test_results = fuzzyshell.search("test", top_k=1)
                            logger.info("ANN index verified ready - first search should be fast")
                        else:
                            logger.warning("ANN index not ready after ingestion")
                    except Exception as e:
                        logger.warning(f"ANN verification failed: {e}")
                        time.sleep(0.5)
                        
                    if no_random:
                        advance_to_step(4, f"Indexed {added_count} commands from history")
                    else:
                        advance_to_step(4, f"Found {added_count} gems in your history!")
                else:
                    if no_random:
                        advance_to_step(4, "No command history found")
                    else:
                        advance_to_step(4, "Your command history is mysteriously empty...")
            else:
                if no_random:
                    advance_to_step(4, f"Database already contains {indexed_count} commands")
                else:
                    advance_to_step(4, "Already know your command secrets!")
            
            advance_to_step(5)
            time.sleep(0.5)
            
            advance_to_step(6)
            
            # Pre-warm main TUI components during final step to reduce transition lag  
            try:
                # Quick pre-initialization of search components (warms up caches/connections)
                fuzzyshell.search("", top_k=1)  # Warm up search engine
            except:
                pass  # Ignore any pre-warming errors
                
            time.sleep(1.2)  # Reduced from 1.5s since pre-warming takes ~0.3s
            
            setup_complete = True
            time.sleep(0.5)
            
        except Exception as e:
            setup_error = f"Setup failed: {str(e)}"
            loop.set_alarm_in(0, lambda loop, user_data: startup_manager.set_error(setup_error))
            time.sleep(1.0)
        
        # Immediate transition to minimize CLI flash  
        loop.set_alarm_in(0.01, lambda loop, data: raise_exit_from_callback())
    
    # Start setup work in background thread
    setup_thread = threading.Thread(target=setup_delegate, daemon=True)
    setup_thread.start()

    def _keep_waking(loop, user_data):
        loop.set_alarm_in(0.5, _keep_waking)

    loop.set_alarm_in(0.5, _keep_waking)

    try:
        loop.run()
    except Exception:
        pass
    
    # Wait for background thread to complete
    setup_thread.join(timeout=2.0)
    
    if setup_error:
        screen_did_become_hidden("OnboardingScreen", "setup_failed")
        stop_tui_redirect()
        set_tui_mode(False)
        print(f"Setup failed: {setup_error}")
        return False
        
    if not setup_complete:
        screen_did_become_hidden("OnboardingScreen", "setup_incomplete")
        stop_tui_redirect()
        set_tui_mode(False)
        print("Setup incomplete")
        return False
    screen_did_become_hidden("OnboardingScreen", "setup_complete")
    
    # Keep TUI mode active during transition to prevent CLI flash
    screen_did_become_visible("MainSearchTUI", {"transition_from": "onboarding"})
    
    # Immediate transition - don't let TUI mode drop
    try:
        result = main_tui_callback(None)
        screen_did_become_hidden("MainSearchTUI", "normal_exit")
        stop_tui_redirect()
        set_tui_mode(False)
        return result
    except Exception as e:
        screen_did_become_hidden("MainSearchTUI", "error")
        stop_tui_redirect()
        set_tui_mode(False)
        print(f"Main TUI failed: {e}")
        return False